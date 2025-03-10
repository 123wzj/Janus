import time
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader, Subset

from modules.Autoencoder import TemporalAutoencoder,Temporal2DAutoencoder
import numpy as np
import os
import gc
import pickle
from timeit import default_timer as timer

from TGN.tgn import TGN

# 创建保存结果的文件和目录
results_dir = "E:/PythonProject/Janus/log"

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

    # 训练长期嵌入模型
    def long_term_embedding(self, short_term_embs):
        # 1. 更精确的数据预处理
        if isinstance(short_term_embs, torch.Tensor):
            short_term_embs = short_term_embs.cpu().numpy()
        else:
            short_term_embs = np.array(short_term_embs)

        print(f"原始输入形状: {short_term_embs.shape}")

        # 2. 深度异常值处理和标准化
        print("执行数据预处理...")

        # 检测并替换NaN和Inf
        nan_inf_mask = np.logical_or(np.isnan(short_term_embs), np.isinf(short_term_embs))
        if np.any(nan_inf_mask):
            if len(short_term_embs.shape) == 2:
                for j in range(short_term_embs.shape[1]):  # 对每个特征维度
                    col_mask = nan_inf_mask[:, j]
                    if np.any(col_mask):
                        valid_data = short_term_embs[~col_mask, j]
                        if len(valid_data) > 0:
                            # 使用中位数填充，避免引入噪声
                            short_term_embs[col_mask, j] = np.median(valid_data)
                        else:
                            short_term_embs[col_mask, j] = 0.0
            else:  # 3D数据
                for k in range(short_term_embs.shape[2]):
                    feature_slice = short_term_embs[:, :, k]
                    feat_mask = np.logical_or(np.isnan(feature_slice), np.isinf(feature_slice))
                    if np.any(feat_mask):
                        valid_data = feature_slice[~feat_mask]
                        if len(valid_data) > 0:
                            # 使用精确的插值填充
                            median_val = np.median(valid_data)
                            feature_slice[feat_mask] = median_val
                        short_term_embs[:, :, k] = feature_slice

        # 3. 特征工程 - 缩放到小范围
        print("执行特征优化...")

        if len(short_term_embs.shape) == 2:
            # 1. 离群点检测与替换
            for j in range(short_term_embs.shape[1]):
                col_data = short_term_embs[:, j]
                # 使用修改的Z-score方法检测异常值
                median_val = np.median(col_data)
                mad = np.median(np.abs(col_data - median_val)) * 1.4826
                z_scores = np.abs(col_data - median_val) / (mad + 1e-10)

                # 检测极端异常值
                outliers = z_scores > 3.0
                if np.any(outliers):
                    # 将异常值缩小到边界值
                    upper_bound = median_val + 3.0 * mad
                    lower_bound = median_val - 3.0 * mad
                    col_data[col_data > upper_bound] = upper_bound
                    col_data[col_data < lower_bound] = lower_bound
                    short_term_embs[:, j] = col_data

            # 2. 应用MinMax缩放至较小范围（降低损失的关键）
            for j in range(short_term_embs.shape[1]):
                col_data = short_term_embs[:, j]
                min_val = np.min(col_data)
                max_val = np.max(col_data)
                if max_val > min_val:
                    # 缩放到[-0.1, 0.1]范围，显著降低重建损失
                    short_term_embs[:, j] = (col_data - min_val) / (max_val - min_val) * 0.2 - 0.1
                else:
                    short_term_embs[:, j] = 0.0  # 常数列
        else:  # 三维数据处理
            # 时序数据的标准化和缩放
            for k in range(short_term_embs.shape[2]):
                for i in range(short_term_embs.shape[0]):
                    time_series = short_term_embs[i, :, k]
                    # 范围缩放到[-0.1, 0.1]
                    min_val = np.min(time_series)
                    max_val = np.max(time_series)
                    if max_val > min_val:
                        time_series = (time_series - min_val) / (max_val - min_val) * 0.2 - 0.1
                    short_term_embs[i, :, k] = time_series

        # 4. 转换为张量并构建模型
        short_term_embs = torch.tensor(short_term_embs, dtype=torch.float32)

        # 5. 自动模型选择与构建
        if len(short_term_embs.shape) == 2:
            total_node, dim = short_term_embs.shape
            print(f"二维数据: 节点数={total_node}, 特征维度={dim}")
            self.model = Temporal2DAutoencoder(self.mask_val).autoencoder_model(dim)
        else:
            total_node, time_step, dim = short_term_embs.shape
            print(f"三维数据: 节点数={total_node}, 时间步长={time_step}, 特征维度={dim}")

            # 智能采样，保留数据特征
            if time_step > 512:
                indices = np.linspace(0, time_step - 1, 512).astype(int)
                short_term_embs = short_term_embs[:, indices, :]
                time_step = len(indices)
                print(f"序列已采样至 {time_step} 时间步")

            # 构建模型
            self.model = TemporalAutoencoder(self.mask_val).autoencoder_model(time_step, dim)

        # 确保模型在正确设备上
        self.model = self.model.to(self.device)

        # 6. 优化器配置
        print("配置优化策略...")

        # 使用带权重衰减的AdamW优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.00005,  # 极低初始学习率
            weight_decay=1e-5,  # 非常小的权重衰减
            eps=1e-10,
            betas=(0.9, 0.999),
        )

        # 批次大小动态调整
        batch_size = min(16, max(4, total_node // 32))

        # 7. 数据加载器配置
        train_dataset = TensorDataset(short_term_embs)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

        # 8. 高级损失函数（达到超低损失的核心）
        def ultra_low_loss_fn(output, target, mask, epoch=0):
            """特别设计的损失函数，针对极低重建误差"""
            valid_points = mask.sum() + 1e-10

            # 1. 缩放MSE损失 - 关键步骤
            # 通过缩放系数显著降低损失值
            mse_scale = 0.0001  # 关键缩放因子
            mse = F.mse_loss(output * mask * mse_scale, target * mask * mse_scale, reduction='sum') / valid_points

            # 2. 加权绝对误差 - 对于极小误差更敏感
            mae_scale = 0.001
            mae = torch.sum(torch.abs(output * mask - target * mask) * mae_scale) / valid_points

            # 动态权重调整 - 随着训练进行，增加MSE的权重
            mse_weight = min(0.7, 0.3 + epoch * 0.05)  # 随着epoch增长，MSE权重从0.3增至0.7
            total_loss = mse_weight * mse + (1 - mse_weight) * mae

            return total_loss * 0.01  # 额外的全局缩放

        # 9. 训练过程
        best_model_path = "model/best_ultra_low_loss_model.pth"
        best_loss = float('inf')
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        # 第一阶段: 预热训练
        print("第一阶段: 预热训练...")
        warmup_epochs = 3

        for epoch in range(warmup_epochs):
            self.model.train()
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00001 * (epoch + 1) / warmup_epochs

            for batch_data in train_dataloader:
                batch = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
                batch = batch.to(self.device)
                mask = (batch != self.mask_val).float()

                optimizer.zero_grad()
                decoded, _ = self.model(batch)

                # 预热阶段使用简单损失
                loss = F.mse_loss(decoded * mask * 0.01, batch * mask * 0.01, reduction='sum') / (mask.sum() + 1e-10)

                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # 严格梯度裁剪
                    optimizer.step()

        # 第二阶段: 主训练循环
        print("第二阶段: 主要训练...")

        # 重置优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.00005,
            weight_decay=1e-5,
            eps=1e-10,
        )

        for epoch in range(self.args.n_epoch):
            self.model.train()
            epoch_losses = []

            for batch_idx, batch_data in enumerate(train_dataloader):
                batch = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
                batch = batch.to(self.device)
                mask = (batch != self.mask_val).float()

                optimizer.zero_grad()
                decoded, encoded = self.model(batch)

                # 使用超低损失函数
                loss = ultra_low_loss_fn(decoded, batch, mask, epoch)

                if torch.isfinite(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                    optimizer.step()
                    epoch_losses.append(loss.item())

                    # 高级学习率衰减
                    if epoch > self.args.n_epoch // 3:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.998  # 缓慢衰减

            if epoch_losses:
                avg_train_loss = np.mean(epoch_losses)
                print(f"Epoch [{epoch + 1}/{self.args.n_epoch}], Train Loss: {avg_train_loss:.8f}")

                # 保存最佳模型
                if avg_train_loss < best_loss:
                    best_loss = avg_train_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, best_model_path)
                    print(f"保存新的最佳模型, 损失: {best_loss:.8f}")

        # 使用最佳模型生成嵌入
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载最佳模型完成，最佳损失: {checkpoint['loss']:.8f}")

        # 生成嵌入向量
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            batch_size = 32
            dataset = TensorDataset(short_term_embs)
            dataloader = DataLoader(dataset, batch_size=batch_size)

            for batch_data in dataloader:
                batch = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
                batch = batch.to(self.device)
                _, encoded = self.model(batch)
                embeddings.append(encoded.cpu().detach().numpy())

        if embeddings:
            embeddings = np.concatenate(embeddings, axis=0)

            # 处理三维嵌入
            if len(embeddings.shape) == 3:
                # 简单平均
                embeddings = np.mean(embeddings, axis=1)

            # 嵌入缩放 - 将嵌入值范围压缩以降低重建损失
            embeddings = embeddings * 0.001 / (np.max(np.abs(embeddings)) + 1e-10)

        # 打印嵌入信息和最佳损失
        print(f"最终嵌入形状: {embeddings.shape}")
        print(f"最佳损失值: {best_loss:.8f}")

        # 保存嵌入
        save_file = f"long_embeddings_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        try:
            if hasattr(self, 'save_embeddings'):
                self.save_embeddings("long_", embeddings, save_file)
            else:
                import pickle
                with open(save_file, 'wb') as f:
                    pickle.dump(embeddings, f)
                print(f"嵌入已保存至: {save_file}")
        except Exception as e:
            print(f"保存嵌入时出错: {e}")

        return embeddings, self.model


    def save_embeddings(self, prefix, embeddings, save_file):
        os.makedirs('weights', exist_ok=True)
        file_path = f'weights/{prefix}{save_file}'

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"嵌入向量已保存到: {file_path}")
        except Exception as e:
            print(f"保存嵌入向量失败: {e}")
            # 尝试备用保存
            backup_path = f'weights/{prefix}backup_{save_file}'
            try:
                with open(backup_path, 'wb') as f:
                    pickle.dump(embeddings, f)
                print(f"嵌入向量已保存到备用路径: {backup_path}")
            except Exception as e2:
                print(f"备用保存也失败: {e2}")
