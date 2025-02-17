import time
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader

from modules.Autoencoder import AutoencoderModel
import numpy as np
import os
import gc
import pickle
from timeit import default_timer as timer

from utils.data_processing import compute_time_statistics, get_data
from utils.utils import get_neighbor_finder
from TGN.tgn import TGN


# 定义自监督
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = timer()

    def end(self, operation_name):
        if self.start_time is None:
            return
        duration = timer() - self.start_time
        print(f"{operation_name} completed in {duration:.2f} seconds")
        self.start_time = None


# 清理内存
def memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 初始化TGN模型
def initialize_tgn_model(args, device, neighbor_finder, node_features, edge_features, time_stats):
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = time_stats

    tgn = TGN(neighbor_finder=neighbor_finder,
              node_features=node_features,
              edge_features=edge_features,
              device=device,
              n_layers=args.n_layers,
              n_heads=args.n_heads,
              dropout=args.dropout,
              use_memory=args.use_memory,
              memory_update_at_start=not args.memory_update_at_end,
              message_dimension=args.message_dimension,
              memory_dimension=args.memory_dimension,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              mean_time_shift_src=mean_time_shift_src,
              std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst,
              std_time_shift_dst=std_time_shift_dst,
              n_neighbors=args.n_degree,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep)

    return tgn.to(device)


def process_nodes_in_batches(nodes, tgn, G, args, device, node_list, batch_size=128):
    """处理节点批次并计算嵌入"""
    max_node_idx = max(G.nodes())
    max_neighbors = args.n_degree
    embedding_dim = args.node_dim

    # 过滤有效节点
    valid_nodes = [n for n in nodes if n <= max_node_idx]

    if not valid_nodes:
        return np.zeros((len(nodes), embedding_dim), dtype=np.float32)

    node_embs = np.zeros((len(nodes), embedding_dim), dtype=np.float32)

    for i in range(0, len(valid_nodes), batch_size):
        batch_nodes = valid_nodes[i:i + batch_size]
        batch_data = []
        valid_batch_nodes = []

        # 收集批次数据
        for node in batch_nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            valid_data = {
                'destinations': [],
                'timestamps': [],
                'edge_idxs': []
            }

            for neighbor in neighbors:
                if neighbor <= max_node_idx:
                    edge_dict = G.get_edge_data(node, neighbor)
                    if edge_dict:
                        first_edge = list(edge_dict.values())[0]
                        valid_data['destinations'].append(neighbor)
                        valid_data['timestamps'].append(first_edge['time'])
                        valid_data['edge_idxs'].append(first_edge['idx'])

                        if len(valid_data['destinations']) >= max_neighbors:
                            break

            if valid_data['destinations']:
                n_pad = max_neighbors - len(valid_data['destinations'])
                if n_pad > 0:
                    valid_data['destinations'].extend([valid_data['destinations'][0]] * n_pad)
                    valid_data['timestamps'].extend([valid_data['timestamps'][0]] * n_pad)
                    valid_data['edge_idxs'].extend([valid_data['edge_idxs'][0]] * n_pad)

                batch_data.append({
                    'source': node,
                    'destinations': valid_data['destinations'][:max_neighbors],
                    'timestamps': valid_data['timestamps'][:max_neighbors],
                    'edge_idxs': valid_data['edge_idxs'][:max_neighbors]
                })
                valid_batch_nodes.append(node)

        if not batch_data:
            continue

        # 修改：节点索引使用长整型，不需要梯度
        source_nodes = torch.tensor([data['source'] for data in batch_data],
                                    device=device,
                                    dtype=torch.long)  # 使用long类型，不需要梯度

        destination_nodes = torch.tensor(
            [d for data in batch_data for d in data['destinations']],
            device=device,
            dtype=torch.long  # 使用long类型，不需要梯度
        )

        # 时间戳使用浮点型，可以计算梯度
        edge_times = torch.tensor(
            [t for data in batch_data for t in data['timestamps']],
            device=device,
            dtype=torch.float32,  # 使用float类型
            requires_grad=True  # 只对浮点型启用梯度
        )

        edge_idxs = torch.tensor(
            [e for data in batch_data for e in data['edge_idxs']],
            device=device,
            dtype=torch.long  # 使用long类型，不需要梯度
        )

        # 负样本节点索引
        valid_node_list = [n for n in node_list if n <= max_node_idx]
        negative_nodes = torch.tensor(
            np.random.choice(valid_node_list, size=len(destination_nodes)),
            device=device,
            dtype=torch.long  # 使用long类型，不需要梯度
        )

        # 计算嵌入
        with torch.set_grad_enabled(tgn.training):
            # 确保输入的张量类型正确
            source_nodes = source_nodes.long()  # 节点索引使用长整型
            destination_nodes = destination_nodes.long()
            negative_nodes = negative_nodes.long()
            edge_times = edge_times.float()  # 时间戳使用浮点型
            edge_idxs = edge_idxs.long()

            source_node_embedding, _, _ = tgn.compute_temporal_embeddings(
                source_nodes=source_nodes,
                destination_nodes=destination_nodes,
                negative_nodes=negative_nodes,
                edge_times=edge_times,
                edge_idxs=edge_idxs,
                n_neighbors=args.n_degree
            )

        # 保存嵌入结果
        embeddings = source_node_embedding.detach().cpu().numpy()
        for node_idx, node in enumerate(valid_batch_nodes):
            original_idx = nodes.index(node)
            if original_idx < len(node_embs):
                node_embs[original_idx] = embeddings[node_idx]

    return node_embs

# 生成短期的图嵌入
def short_term_embedding(args, node_list, idx, G):
    print(f"Processing snapshot {idx} with {len(node_list)} nodes")
    node_features, edge_features, full_data = get_data(G.snapshot_idx, args)
    neighbor_finder = get_neighbor_finder(G, args.uniform)
    time_stats = compute_time_statistics(G.sources, G.destinations, G.timestamps)
    tgn = initialize_tgn_model(args, args.device, neighbor_finder, node_features, edge_features, time_stats)
    node_embs = process_nodes_in_batches(node_list, tgn, G, args, args.device, node_list)
    return node_embs


class Janus:
    def __init__(self, args, device, node_list, node_map, graphs):
        self.args = args
        self.mask_val = 0.
        self.node_list = node_list
        self.node_map = node_map
        self.graphs = graphs
        self.model, self.encoder = None, None
        self.device = device
        self.monitor = PerformanceMonitor()
        self.create_directories()

    @staticmethod
    def create_directories():
        directories = ['weights', 'model', 'fig']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    # 训练长期嵌入模型
    def long_term_embedding(self, short_term_embs):

        # 详细的数据分析
        print("\n输入数据分析...")
        short_term_embs = np.array(short_term_embs)
        total_elements = short_term_embs.size
        nan_mask = np.isnan(short_term_embs)
        inf_mask = np.isinf(short_term_embs)

        print(f"数据统计:")
        print(f"- 总元素数: {total_elements}")
        print(f"- NaN数量: {np.sum(nan_mask)} ({np.sum(nan_mask) / total_elements * 100:.2f}%)")
        print(f"- Inf数量: {np.sum(inf_mask)}")

        # 分析NaN的分布
        nan_by_node = np.sum(nan_mask, axis=(1, 2))
        nan_by_time = np.sum(nan_mask, axis=(0, 2))
        nan_by_dim = np.sum(nan_mask, axis=(0, 1))

        print("\nNaN分布分析:")
        print(f"- 包含NaN的节点数: {np.sum(nan_by_node > 0)}")
        print(f"- 包含NaN的时间步数: {np.sum(nan_by_time > 0)}")
        print(f"- 包含NaN的特征维度数: {np.sum(nan_by_dim > 0)}")

        # 数据修复策略
        print("\n应用数据修复策略...")

        # 1. 使用临近值填充
        valid_mask = ~nan_mask
        for i in range(short_term_embs.shape[0]):  # 对每个节点
            for j in range(short_term_embs.shape[2]):  # 对每个特征维度
                node_feat = short_term_embs[i, :, j]
                if np.all(np.isnan(node_feat)):
                    # 如果整个序列都是NaN，填充0
                    short_term_embs[i, :, j] = 0
                else:
                    # 使用临近有效值填充
                    valid_indices = np.where(~np.isnan(node_feat))[0]
                    if len(valid_indices) > 0:
                        for k in range(len(node_feat)):
                            if np.isnan(node_feat[k]):
                                # 找到最近的有效值
                                if len(valid_indices) == 0:
                                    node_feat[k] = 0
                                else:
                                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - k))]
                                    node_feat[k] = node_feat[nearest_idx]
                        short_term_embs[i, :, j] = node_feat

        # 2. 标准化有效数据
        print("\n数据标准化...")
        valid_data = short_term_embs[~np.isnan(short_term_embs)]
        if len(valid_data) > 0:
            mean = np.mean(valid_data)
            std = np.std(valid_data)
            if std > 0:
                short_term_embs = (short_term_embs - mean) / (std + 1e-8)

        print(f"数据范围: [{np.min(short_term_embs):.4f}, {np.max(short_term_embs):.4f}]")

        # 转换为张量
        short_term_embs = torch.tensor(short_term_embs, dtype=torch.float32)
        total_node, time_step, dim = short_term_embs.shape

        print(f"\n数据维度:")
        print(f"- 节点数: {total_node}")
        print(f"- 时间步长: {time_step}")
        print(f"- 特征维度: {dim}")

        # 模型初始化
        self.model = AutoencoderModel(self.mask_val).autoencoder_model(time_step, dim)
        self.model = self.model.to(self.device)

        # 使用更保守的优化器设置
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=0.001,
            eps=1e-8
        )

        # 余弦退火学习率
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # 自定义损失函数
        def robust_loss(output, target):
            """具有鲁棒性的损失函数"""
            diff = output - target
            # Huber损失
            delta = 1.0
            abs_diff = torch.abs(diff)
            quadratic = torch.min(abs_diff, torch.tensor(delta).to(self.device))
            linear = abs_diff - quadratic
            loss = 0.5 * quadratic.pow(2) + delta * linear
            return torch.mean(loss)

        print("\n开始训练...")
        best_loss = float('inf')
        loss_history = []

        # 使用较小的批次大小
        batch_size = min(16, total_node)
        dataset = TensorDataset(short_term_embs)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        for epoch in range(self.args.n_epoch):
            self.model.train()
            epoch_losses = []

            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device)

                optimizer.zero_grad()
                decoded, encoded = self.model(batch)

                loss = robust_loss(decoded, batch)

                if torch.isfinite(loss):
                    loss.backward()
                    # 更严格的梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=0.1
                    )
                    optimizer.step()
                    epoch_losses.append(loss.item())

            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                loss_history.append(avg_loss)
                scheduler.step()

                print(f'Epoch [{epoch}/{self.args.n_epoch}], '
                      f'Loss: {avg_loss:.4f}, '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save_best_model()

        # 生成长期嵌入
        print("\n生成嵌入...")
        self.model.eval()
        with torch.no_grad():
            embeddings = []
            for batch in DataLoader(dataset, batch_size=32):
                batch = batch[0].to(self.device)
                _, encoded = self.model(batch)
                embeddings.append(encoded.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        embeddings = embeddings.transpose(1, 0, 2)

        # 还原数据范围
        if 'std' in locals() and std > 0:
            embeddings = embeddings * (std + 1e-8) + mean

        print(f"\n训练完成:")
        print(f"- 最终损失: {loss_history[-1]:.4f}")
        print(f"- 最佳损失: {best_loss:.4f}")
        print(f"- 嵌入范围: [{np.min(embeddings):.4f}, {np.max(embeddings):.4f}]")

        return embeddings

    # 保存最佳模型
    def save_best_model(self):
        save_path = 'model/best_autoencoder.pth'
        os.makedirs('model', exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict()
        }, save_path)

    # 保存训练结果，包括损失曲线和模型参数
    def save_training_results(self, loss_history):
        # 确保目录存在
        os.makedirs("fig", exist_ok=True)
        os.makedirs("model", exist_ok=True)

        # 绘制并保存损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig("fig/train_error.png")
        plt.close()

        # 保存完整模型
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'loss_history': loss_history
        }, 'model/autoencoder.pth')

        # 保存编码器部分
        torch.save({
            'encoder_state_dict': self.model.encoder.state_dict(),
        }, 'model/encoder.pth')


    def save_embeddings(self, prefix, embeddings, save_file):
        file_path = f'weights/{prefix}{save_file}'
        with open(file_path, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Saved {prefix} embeddings to {file_path}")

    def generate_short_term_embeddings(self, save_file):
        print("Generating Short Term embeddings...")
        num_cores = min(4, os.cpu_count())
        print(f"Using {num_cores} CPU cores")

        data_tuple = [(self.args, self.node_list, idx, G)
                      for idx, G in enumerate(self.graphs)]

        self.monitor.start()
        with Pool(num_cores) as pool:
            short_term_embs = pool.starmap(short_term_embedding, data_tuple)
        self.monitor.end("Short Term embedding generation")

        short_term_embs = np.array(short_term_embs)
        self.save_embeddings('short_term', short_term_embs, save_file)
        return short_term_embs

    def generate_long_term_embeddings(self, short_term_embs, save_file):
        print("Starting Long Term Embedding generation...")
        short_term_embs = np.transpose(short_term_embs, (1, 0, 2))

        self.monitor.start()
        dynamic_embs = self.long_term_embedding(short_term_embs)
        self.monitor.end("Long Term Embedding generation")

        self.save_embeddings('long_term', dynamic_embs, save_file)
        return dynamic_embs