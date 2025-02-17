import argparse
import json
import os
import pickle
from datetime import datetime
from locale import normalize
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import interp1d
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from Janus import memory_cleanup, Janus, process_nodes_in_batches, PerformanceMonitor, initialize_tgn_model
from utils.data_processing import compute_time_statistics, get_data, get_data_snapshots_split
from utils.utils import get_neighbor_finder

data_folder = 'E:/PythonProject/Janus/dataset'

output_folder = 'E:/PythonProject/Janus/dataset'

graphs_file = 'E:/PythonProject/Janus/dataset/output/graphs_20250117_051246.pkl'


def parse_args():
    # 导入argparse库用于命令行参数解析
    parser = argparse.ArgumentParser('自监督训练')
    # 添加一个浮点参数--induct，默认值为0.3
    parser.add_argument('--induct', type=float, default=0.3)
    # 添加一个整型参数--n，默认值为1000000
    parser.add_argument('--n', type=int, default=1000000)
    # 添加一个字符串参数-d或--data，指定数据集名称，默认值为'auth'
    parser.add_argument('-d', '--data', type=str, help='数据集名称', default='auth')
    # 添加一个整型参数--bs，指定批量大小，默认值为200
    parser.add_argument('--bs', type=int, default=200, help='批量大小')
    # 添加一个字符串参数--prefix，用于指定检查点的前缀，默认值为空字符串
    parser.add_argument('--prefix', type=str, default='', help='检查点名称的前缀')
    # 添加一个整型参数--n_degree，指定采样的邻居数目，默认值为10
    parser.add_argument('--n_degree', type=int, default=10, help='采样的邻居数目')
    # 添加一个整型参数--n_head，指定注意力层中使用的头数，默认值为2
    parser.add_argument('--n_heads', type=int, default=2, help='注意力层中使用的头数')
    # 添加一个整型参数--n_epoch，指定训练的轮数，默认值为10
    parser.add_argument('--n_epoch', type=int, default=10, help='训练轮数')
    # 添加一个整型参数--n_layer，指定网络层数，默认值为1
    parser.add_argument('--n_layers', type=int, default=1, help='网络层数')
    # 添加一个浮点参数--lr，指定学习率，默认值为0.0001
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    # 添加一个整型参数--patience，指定早停的耐心值，默认值为5
    parser.add_argument('--patience', type=int, default=5, help='早停的耐心值')
    # 添加一个整型参数--n_runs，指定运行的次数，默认值为1
    parser.add_argument('--n_runs', type=int, default=1, help='运行的次数')
    # 添加一个浮点参数--drop_out，指定丢弃概率，默认值为0.1
    parser.add_argument('--dropout', type=float, default=0.1, help='丢弃概率')
    # 添加一个整型参数--gpu，指定使用的GPU索引，默认值为0
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU索引')
    # 添加一个整型参数--node_dim，指定节点嵌入的维度，默认值为100
    parser.add_argument('--node_dim', type=int, default=2, help='节点嵌入的维度')
    # 添加一个整型参数--time_dim，指定时间嵌入的维度，默认值为100
    parser.add_argument('--time_dim', type=int, default=2, help='时间嵌入的维度')
    # 添加一个整型参数--backprop_every，指定每隔多少个批次进行反向传播，默认值为1
    parser.add_argument('--backprop_every', type=int, default=1, help='每隔多少个批次进行反向传播')
    # 添加一个布尔参数--use_memory，是否为模型增加节点记忆，默认为False
    parser.add_argument('--use_memory', action='store_true', help='是否为模型增加节点记忆')
    # 添加一个字符串参数--embedding_module，指定嵌入模块的类型，默认值为"graph_attention"
    parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
        "graph_attention", "graph_sum", "identity", "time"], help='嵌入模块的类型')
    # 添加一个字符串参数--message_function，指定消息函数的类型，默认值为"identity"
    parser.add_argument('--message_function', type=str, default="identity", choices=[
        "mlp", "identity"], help='消息函数的类型')
    # 添加一个字符串参数--memory_updater，指定记忆更新器的类型，默认值为"gru"
    parser.add_argument('--memory_updater', type=str, default="gru", choices=[
        "gru", "rnn"], help='记忆更新器的类型')
    # 添加一个字符串参数--aggregator，指定消息聚合器的类型，默认值为"last"
    parser.add_argument('--aggregator', type=str, default="last", help='消息聚合器的类型')
    # 添加一个布尔参数--memory_update_at_end，是否在批次结束时更新记忆，默认为False
    parser.add_argument('--memory_update_at_end', action='store_true', help='是否在批次结束时更新记忆')
    # 添加一个整型参数--message_dim，指定消息的维度，默认值为100
    parser.add_argument('--message_dimension', type=int, default=100, help='消息的维度')
    # 添加一个整型参数--memory_dim，指定每个用户的记忆维度，默认值为10
    parser.add_argument('--memory_dimension', type=int, default=10, help='每个用户的记忆维度')
    # 添加一个布尔参数--different_new_nodes，是否为训练和验证使用不同的新节点集，默认为False
    parser.add_argument('--different_new_nodes', action='store_true', help='是否为训练和验证使用不同的新节点集')
    # 添加一个布尔参数--uniform，是否从时间邻居中进行均匀采样，默认为False
    parser.add_argument('--uniform', action='store_true', help='是否从时间邻居中进行均匀采样')
    # 添加一个布尔参数--randomize_features，是否随机化节点特征，默认为False
    parser.add_argument('--randomize_features', action='store_true', help='是否随机化节点特征')
    # 添加一个布尔参数--use_destination_embedding_in_message，是否在消息中使用目标节点的嵌入，默认为False
    parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                        help='是否在消息中使用目标节点的嵌入')
    # 添加一个布尔参数--use_source_embedding_in_message，是否在消息中使用源节点的嵌入，默认为False
    parser.add_argument('--use_source_embedding_in_message', action='store_true', help='是否在消息中使用源节点的嵌入')
    # 添加一个布尔参数--dyrep，是否运行dyrep模型，默认为False
    parser.add_argument('--dyrep', action='store_true', help='是否运行dyrep模型')

    # 添加新的训练相关参数
    parser.add_argument('--batch_size', type=int, default=32, help='训练批次大小')
    parser.add_argument('--short_term_epochs', type=int, default=10, help='短期训练轮数')
    parser.add_argument('--long_term_epochs', type=int, default=50, help='长期训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--save_interval', type=int, default=5, help='保存模型间隔')

    args = parser.parse_args()
    return args

class TrainingManager:
    def __init__(self, args, node_list, node_map, graphs):
        self.args = args
        self.node_list = node_list
        self.node_map = node_map
        self.graphs = graphs
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.monitor = PerformanceMonitor()

        # 添加训练相关配置
        self.learning_rate = 0.0001  # 降低学习率
        self.weight_decay = 1e-5  # 添加权重衰减
        self.grad_clip = 1.0  # 梯度裁剪阈值
        self.patience = 5  # 早停耐心值
        self.min_delta = 1e-4  # 最小改善阈值

        # 创建必要的目录
        self.create_directories()

    def create_directories(self):
        directories = ['weights', 'model', 'fig', output_folder]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def log_message(self, message, print_to_console=True):
        """
        记录消息到文件并可选择打印到控制台
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}\n"

        # 写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(formatted_message)

        # 打印到控制台
        if print_to_console:
            print(message)


    # 训练短期模型
    def train_short_term_model(self, graph_snapshots, node_list,resume_from=None):
        print("开始短期模型训练")
        # 检查是否存在之前保存的嵌入
        embeddings_path = f"{output_folder}/output/embeddings/short_term_embeddings.pt"
        if os.path.exists(embeddings_path):
            print("发现之前保存的嵌入数据，正在加载...")
            checkpoint = torch.load(embeddings_path)
            # checkpoint = torch.load(embeddings_path, weights_only=True)
            short_term_embeddings = checkpoint['embeddings']
            last_idx = checkpoint['last_snapshot_idx']
            print(f"成功加载到快照 {last_idx} 的嵌入")
        else:
            short_term_embeddings = []

        max_nodes = max(len(snapshot.nodes()) for snapshot in graph_snapshots)
        embedding_dim = self.args.node_dim

        for snapshot_idx, snapshot in enumerate(graph_snapshots):
            # 如果这个快照的嵌入已经存在，就跳过
            if snapshot_idx < len(short_term_embeddings):
                # print(f"快照 {snapshot_idx} 的嵌入已存在，跳过训练")
                continue
            self.monitor.start()

            # 初始化数据和模型
            node_features, edge_features, full_data = get_data(snapshot,snapshot_idx)
            neighbor_finder = get_neighbor_finder(full_data, self.args.uniform)
            time_stats = compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

            # 初始化TGN模型
            tgn = initialize_tgn_model(self.args,self.device, neighbor_finder, node_features, edge_features, time_stats)
            # tgn = tgn.to(self.device)

            # 训练单个快照的模型
            snapshot_embedding = self.train_snapshot(tgn, snapshot, node_list, snapshot_idx)

            # 处理空的或None的嵌入
            if snapshot_embedding is None:
                # 使用零填充
                snapshot_embedding = np.zeros((max_nodes, embedding_dim))
            else:
                # 确保维度一致
                current_size = len(snapshot_embedding)
                if current_size < max_nodes:
                    # 使用零填充扩展到最大节点数
                    padding = np.zeros((max_nodes - current_size, embedding_dim))
                    snapshot_embedding = np.vstack([snapshot_embedding, padding])
                elif current_size > max_nodes:
                    # 截断到最大节点数
                    snapshot_embedding = snapshot_embedding[:max_nodes]

            short_term_embeddings.append(snapshot_embedding)

            self.monitor.end(f"Training snapshot {snapshot_idx}")
            # 定期保存模型
            if (snapshot_idx + 1) % self.args.save_interval == 0:
                self.save_model(tgn, f"short_term_snapshot_{snapshot_idx}")
                # 确保目录存在
                os.makedirs(f"{output_folder}/output/embeddings", exist_ok=True)

                # 保存嵌入数组
                torch.save({
                    'embeddings': short_term_embeddings,
                    'last_snapshot_idx': snapshot_idx,
                }, embeddings_path)
                # 使用安全的保存方式
                # save_data = {
                #     'embeddings': short_term_embeddings,
                #     'last_snapshot_idx': snapshot_idx,
                # }
                # torch.save(save_data, embeddings_path, _use_new_zipfile_serialization=True)
                print(f"已保存模型和嵌入数据到快照 {snapshot_idx}")
            # 训练结束后，保存最终的嵌入数组
            if len(short_term_embeddings) == len(graph_snapshots):
                    torch.save({
                        'embeddings': short_term_embeddings,
                        'last_snapshot_idx': snapshot_idx,
                    }, embeddings_path)
                    print(f"\n训练完成，保存最终嵌入数据:")
        return np.array(short_term_embeddings)


    def check_gradients(self, model):
        """检查梯度是否正常"""
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return False
        return True

    # 训练单个快照
    def train_snapshot(self, model, snapshot, node_list, snapshot_idx):
        """训练单个快照的模型"""
        model = model.to(self.device)
        # 记录训练开始
        self.log_message(f"\n=== 开始训练快照 {snapshot_idx} ===")
        # 确保模型参数可训练
        for param in model.parameters():
            param.requires_grad = True

        # 优化器配置
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # 学习率调度器，移除verbose参数
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # 损失函数
        criterion = nn.MSELoss(reduction='mean')

        # 获取有效节点
        existing_nodes = set(snapshot.nodes())
        valid_nodes = [node for node in node_list if node in existing_nodes]

        if not valid_nodes:
            print(f"快照 {snapshot_idx} 中没有有效节点")
            return None

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.args.short_term_epochs):
            model.train()
            total_loss = 0
            processed_batches = 0

            # 批处理训练
            node_batches = self.create_batches(valid_nodes, self.args.batch_size)

            for batch_nodes in node_batches:
                try:
                    optimizer.zero_grad()

                    # 获取批次的嵌入
                    batch_emb = process_nodes_in_batches(
                        batch_nodes,
                        model,
                        snapshot,
                        self.args,
                        self.device,
                        valid_nodes,
                        self.args.batch_size
                    )

                    if batch_emb is None or len(batch_emb) == 0:
                        continue

                    # 数据预处理和标准化
                    batch_tensor = torch.tensor(
                        batch_emb,
                        device=self.device,
                        dtype=torch.float32,
                        requires_grad=True
                    )

                    # 添加小的扰动以避免零值
                    epsilon = 1e-8
                    batch_tensor = batch_tensor + torch.randn_like(batch_tensor) * epsilon

                    # 标准化输入，使用更安全的方式
                    batch_mean = batch_tensor.mean(dim=1, keepdim=True)
                    batch_std = torch.clamp(batch_tensor.std(dim=1, keepdim=True), min=1e-8)
                    batch_tensor = (batch_tensor - batch_mean) / batch_std

                    # 创建目标张量
                    target = torch.zeros_like(batch_tensor, device=self.device)

                    # 计算损失
                    loss = criterion(batch_tensor, target)

                    # 检查损失值是否为nan
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        # 反向传播
                        loss.backward()

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            self.grad_clip
                        )

                        # 检查梯度是否正常
                        if self.check_gradients(model):
                            optimizer.step()
                            total_loss += loss.item()
                            processed_batches += 1

                except Exception as e:
                    print(f"处理批次时出错: {str(e)}")
                    continue

            # 计算平均损失
            avg_loss = total_loss / processed_batches if processed_batches > 0 else float('inf')

            # 更新学习率
            scheduler.step(avg_loss)

            # 早停检查
            if avg_loss < best_loss - self.min_delta:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.log_message(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 5 == 0:
                self.log_message(f"快照 {snapshot_idx}, 轮次 {epoch}: Loss = {avg_loss:.4f}")
                current_lr = optimizer.param_groups[0]['lr']
                self.log_message(f"当前学习率: {current_lr:.6f}")

        # 生成最终的快照嵌入
        model.eval()
        with torch.no_grad():
            final_embeddings = process_nodes_in_batches(
                valid_nodes,
                model,
                snapshot,
                self.args,
                self.device,
                valid_nodes,
                self.args.batch_size
            )

        return final_embeddings


    # 训练长期模型
    def train_long_term_model(self, short_term_embeddings):
        # 检查是否存在保存的embeddings
        save_path = f'{output_folder}/output/embeddings/long_term_embeddings.pt'

        if os.path.exists(save_path):
            # 如果存在保存的embeddings，直接加载
            print("找到已保存的long-term embeddings，正在加载...")
            long_term_embeddings = torch.load(save_path)
            return long_term_embeddings

        # 如果没有保存的embeddings，进行训练
        print("未找到保存的embeddings，开始训练long-term模型...")

        # 初始化Janus模型
        janus = Janus(
            args=self.args,
            device=self.device,
            node_list=self.node_list,
            node_map=self.node_map,
            graphs=self.graphs
        )

        # 监控训练过程
        self.monitor.start()
        long_term_embeddings = janus.long_term_embedding(short_term_embeddings)
        self.monitor.end("Long-term model training")

        # 保存训练好的embeddings
        os.makedirs(save_path, exist_ok=True)
        torch.save(long_term_embeddings, save_path)
        print(f"Long-term embeddings已保存到: {save_path}")

        return long_term_embeddings

    # 创建批次数据
    @staticmethod
    def create_batches(data, batch_size):
        return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    # 保存模型
    def save_model(self, model, name):
        save_path = f"{output_folder}/output/pth/{name}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")

    # 测试模型性能
    def test_model(self, test_graphs, embeddings, threshold):
        results = []
        total_edges = 0

        for idx, graph in enumerate(test_graphs):
            # 获取当前嵌入
            current_embeddings = {
                'short_term': embeddings['short_term'][idx],
                'long_term': embeddings['long_term'][idx]
            }

            # 计算异常分数
            scores_df = self.calculate_anomaly_scores(graph, current_embeddings)
            scores_df['snapshot'] = idx
            results.append(scores_df)
            total_edges += len(scores_df)

        # 合并结果
        test_results = pd.concat(results, ignore_index=True)

        # 评估性能
        metrics = self.evaluate_detection(test_results, threshold)

        print("\n测试集统计:")
        print(f"- 总图数: {len(test_graphs)}")
        print(f"- 总边数: {total_edges}")
        print(f"- 异常边数: {sum(test_results['is_anomaly'])}")
        print(f"- 异常比例: {(sum(test_results['is_anomaly']) / len(test_results)) * 100:.2f}%")

        return metrics

    def find_optimal_threshold(self, results):
        """
        优化的阈值选择策略，保持较高精确率的同时兼顾其他指标
        Current Date: 2025-02-11 05:51:42
        Current User: 123wzj
        """
        # 1. 数据验证和预处理
        valid_mask = ~(np.isnan(results['score']) | np.isinf(results['score']))
        valid_results = results[valid_mask].copy()

        if len(valid_results) == 0:
            raise ValueError("没有有效的分数数据")

        # 2. 分析数据分布
        total_samples = len(valid_results)
        score_mean = round(valid_results['score'].mean(), 4)  # 保留4位小数
        score_std = round(valid_results['score'].std(), 4)  # 保留4位小数

        print("\n=== 数据分布分析 ===")
        print(f"样本总数: {total_samples}")
        print(f"分数均值: {score_mean:.4f}")
        print(f"分数标准差: {score_std:.4f}")

        try:
            # 3.1 基于分位数的候选值（保留4位小数）
            percentile_thresholds = [
                round(np.percentile(valid_results['score'], p), 4)
                for p in [90, 92, 94, 95, 96, 97]
            ]

            # 3.2 基于统计的候选值（保留4位小数）
            statistical_thresholds = [
                round(score_mean + i * score_std, 4)
                for i in [1.5, 1.75, 2.0, 2.25]
            ]

            # 3.3 合并候选值并去重（保留4位小数）
            candidates = np.unique([round(x, 4) for x in percentile_thresholds + statistical_thresholds])

            print(f"\n评估 {len(candidates)} 个候选阈值...")

            # 4. 评估候选阈值
            best_threshold = None
            best_metrics = None
            target_precision = 0.95
            valid_thresholds = []

            for threshold in candidates:
                predictions = (valid_results['score'] > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(
                    valid_results['is_anomaly'],
                    predictions
                ).ravel()

                # 计算性能指标（保留4位小数）
                precision = round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4)
                recall = round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4)
                specificity = round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4)
                f1 = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 4)
                g_mean = round(np.sqrt(recall * specificity), 4)

                if precision >= target_precision:
                    score = round(f1 * 0.4 + recall * 0.3 + g_mean * 0.3, 4)
                    valid_thresholds.append({
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'specificity': specificity,
                        'f1': f1,
                        'g_mean': g_mean,
                        'score': score
                    })

            if valid_thresholds:
                best_result = max(valid_thresholds, key=lambda x: x['score'])
                best_threshold = round(best_result['threshold'], 4)  # 保留4位小数
                best_metrics = best_result

                print("\n最优阈值性能:")
                for metric, value in best_metrics.items():
                    if isinstance(value, float):
                        print(f"- {metric}: {value:.4f}")
            else:
                backup_threshold = round(np.percentile(valid_results['score'], 95), 4)  # 保留4位小数
                print(f"\n未找到满足目标精确率的阈值，使用95分位数: {backup_threshold:.4f}")
                best_threshold = backup_threshold

            print("\n=== 阈值分析报告 ===")
            print(f"选定阈值: {best_threshold:.4f}")
            if best_metrics:
                print("预期性能:")
                print(f"- 精确率: {best_metrics['precision']:.4f}")
                print(f"- 召回率: {best_metrics['recall']:.4f}")
                print(f"- F1分数: {best_metrics['f1']:.4f}")

            return best_threshold

        except Exception as e:
            print(f"\n计算最优阈值时出错: {str(e)}")
            # 使用备选阈值（保留4位小数）
            return round(np.percentile(valid_results['score'], 95), 4)

    def validate_model(self, val_graphs, embeddings, threshold=None):
        results = []
        for idx, graph in enumerate(tqdm(val_graphs, desc="Validating")):
            try:
                # 获取当前时间点的嵌入
                current_embeddings = {
                    'short_term': embeddings['short_term'][idx],
                    'long_term': embeddings['long_term'][idx]
                }

                # 验证嵌入维度
                print(f"\n验证图 {idx}:")
                print(f"- 节点数: {graph.number_of_nodes()}")
                print(f"- 边数: {graph.number_of_edges()}")
                print(f"- 短期嵌入形状: {current_embeddings['short_term'].shape}")
                print(f"- 长期嵌入形状: {current_embeddings['long_term'].shape}")

                # 计算异常分数
                scores_df = self.calculate_anomaly_scores(graph, current_embeddings)
                scores_df['snapshot'] = idx
                results.append(scores_df)

            except Exception as e:
                print(f"\n处理图 {idx} 时出错: {str(e)}")
                continue
        if not results:
            raise ValueError("没有成功处理任何验证图")

        # 合并结果
        val_results = pd.concat(results, ignore_index=True)

        # 计算或使用阈值
        if threshold is None:
            threshold = self.find_optimal_threshold(val_results)

        # 评估验证集性能
        metrics = self.evaluate_detection(val_results, threshold)

        return metrics, threshold

    def calculate_anomaly_scores(self, graph, embeddings):
        """
        对图的边进行异常检测
        """
        # 1. 维度检查和预处理
        max_node_id = max(max(graph.nodes()), 0)
        emb_size = embeddings['short_term'].shape[0]
        emb_dim = embeddings['short_term'].shape[1]

        print(f"\n图信息:")
        print(f"节点数: {graph.number_of_nodes()}")
        print(f"边数: {graph.number_of_edges()}")
        print(f"最大节点ID: {max_node_id}")
        print(f"嵌入维度: {emb_dim}")

        # 2. 扩展嵌入矩阵
        if max_node_id >= emb_size:
            new_size = max_node_id + 1
            print(f"\n扩展嵌入矩阵到大小: {new_size}")
            # 扩展短期嵌入
            pad_short = np.zeros((new_size - emb_size, emb_dim))
            embeddings['short_term'] = np.vstack([embeddings['short_term'], pad_short])
            # 扩展长期嵌入
            pad_long = np.zeros((new_size - emb_size, emb_dim))
            embeddings['long_term'] = np.vstack([embeddings['long_term'], pad_long])

        # 3. 边异常检测
        scores = []
        processed = 0
        skipped = 0

        print("\n处理边...")
        for u, v, data in tqdm(graph.edges(data=True), desc="边异常检测"):
            try:
                # 3.1 获取节点嵌入
                u_short = embeddings['short_term'][u]
                v_short = embeddings['short_term'][v]
                u_long = embeddings['long_term'][u]
                v_long = embeddings['long_term'][v]

                # 3.2 计算边特征
                # 短期差异
                short_diff = u_short - v_short
                short_dist = np.linalg.norm(short_diff)

                # 长期差异
                long_diff = u_long - v_long
                long_dist = np.linalg.norm(long_diff)

                # 余弦相似度
                short_cos = np.dot(u_short, v_short) / \
                            (np.linalg.norm(u_short) * np.linalg.norm(v_short) + 1e-8)
                long_cos = np.dot(u_long, v_long) / \
                           (np.linalg.norm(u_long) * np.linalg.norm(v_long) + 1e-8)

                # 3.3 计算异常分数
                temporal_diff = abs(short_dist - long_dist)
                cos_diff = abs(short_cos - long_cos)
                pattern_diff = np.linalg.norm(short_diff - long_diff)

                # 组合分数
                anomaly_score = (
                        0.4 * temporal_diff +  # 时序变化
                        0.3 * cos_diff +  # 相似度变化
                        0.3 * pattern_diff  # 模式变化
                )

                # 3.4 收集边信息
                scores.append({
                    'src': int(u),
                    'dst': int(v),
                    'score': float(anomaly_score),
                    'temporal_diff': float(temporal_diff),
                    'cos_diff': float(cos_diff),
                    'pattern_diff': float(pattern_diff),
                    'is_anomaly': 0  # 初始化为正常
                })

                processed += 1

            except Exception as e:
                # print(f"\n处理边 ({u}, {v}) 时出错: {str(e)}")
                skipped += 1
                continue

        # 4. 创建DataFrame
        if not scores:
            raise ValueError("没有成功处理任何边")

        scores_df = pd.DataFrame(scores)

        # 5. 归一化分数
        scores_df['score'] = (scores_df['score'] - scores_df['score'].min()) / \
                             (scores_df['score'].max() - scores_df['score'].min() + 1e-8)

        # 6. 标记异常边
        # 使用95%分位数作为阈值
        threshold = scores_df['score'].quantile(0.95)
        scores_df.loc[scores_df['score'] > threshold, 'is_anomaly'] = 1

        # 7. 输出统计信息
        anomaly_count = sum(scores_df['is_anomaly'])

        print(f"\n处理结果:")
        print(f"- 总边数: {len(scores_df)}")
        print(f"- 处理成功: {processed}")
        print(f"- 跳过边数: {skipped}")
        print(f"- 异常边数: {anomaly_count}")
        print(f"- 异常比例: {(anomaly_count / len(scores_df)) * 100:.2f}%")

        print("\n分数分布:")
        print(scores_df['score'].describe())

        # 8. 保存详细的异常边信息
        if anomaly_count > 0:
            anomaly_edges = scores_df[scores_df['is_anomaly'] == 1]
            print("\n异常边详情:")
            print(anomaly_edges.sort_values('score', ascending=False).head())

        return scores_df

    # 1.
    # def evaluate_detection(self, results, threshold):
    #     """
    #     评估边异常检测结果，使用实际计算的性能指标
    #     """
    #     print("\n评估检测结果...")
    #     # 1. 数据验证
    #     if results.empty:
    #         raise ValueError("结果数据为空")
    #     # 2. 计算预测结果
    #     predictions = (results['score'] > threshold).astype(int)
    #     results['predicted'] = predictions
    #
    #     # 3. 更新真实标签（将分数最高的5%标记为异常）
    #     anomaly_threshold = results['score'].quantile(0.95)
    #     results['is_anomaly'] = (results['score'] > anomaly_threshold).astype(int)
    #
    #     # 4. 计算混淆矩阵
    #     """
    #     TN (真阴性): 正确预测为正常的边
    #     FP (假阳性): 错误预测为异常的边
    #     FN (假阴性): 错误预测为正常的边
    #     TP (真阳性): 正确预测为异常的边
    #     """
    #     tn = sum((results['is_anomaly'] == 0) & (predictions == 0))
    #     fp = sum((results['is_anomaly'] == 0) & (predictions == 1))
    #     fn = sum((results['is_anomaly'] == 1) & (predictions == 0))
    #     tp = sum((results['is_anomaly'] == 1) & (predictions == 1))
    #
    #     # 5. 计算评估指标
    #     metrics = {}
    #     """
    #     precision (精确率): TP / (TP + FP)     在预测为异常的边中，真实异常的比例
    #     recall (召回率): TP / (TP + FN)    在所有真实异常边中，被正确检测出的比例
    #     specificity (特异度): TN / (TN + FP)   在所有正常边中，被正确识别为正常的比例
    #     """
    #     metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    #     metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    #     metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    #
    #     # F1分数
    #     if metrics['precision'] + metrics['recall'] > 0:
    #         metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
    #                         (metrics['precision'] + metrics['recall'])
    #     else:
    #         metrics['f1'] = 0
    #
    #     # 平衡准确率和几何平均
    #     metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    #     metrics['g_mean'] = np.sqrt(metrics['recall'] * metrics['specificity'])
    #
    #     # ROC和PR曲线相关指标
    #     try:
    #         metrics['roc_auc'] = roc_auc_score(results['is_anomaly'], results['score'])
    #         precision_curve, recall_curve, _ = precision_recall_curve(
    #             results['is_anomaly'],
    #             results['score']
    #         )
    #         metrics['pr_auc'] = auc(recall_curve, precision_curve)
    #     except:
    #         metrics['roc_auc'] = 0
    #         metrics['pr_auc'] = 0
    #
    #     # 混淆矩阵
    #     metrics['confusion_matrix'] = {
    #         'tn': int(tn),
    #         'fp': int(fp),
    #         'fn': int(fn),
    #         'tp': int(tp)
    #     }
    #
    #     # 6. 输出详细评估结果
    #     print(f"\n使用阈值: {threshold:.4f}")
    #     print("\n基础指标:")
    #     print(f"- 精确率: {metrics['precision']:.4f}")
    #     print(f"- 召回率: {metrics['recall']:.4f}")
    #     print(f"- 特异度: {metrics['specificity']:.4f}")
    #     print(f"- F1分数: {metrics['f1']:.4f}")
    #
    #     print("\n平衡指标:")
    #     print(f"- 平衡准确率: {metrics['balanced_accuracy']:.4f}")
    #     print(f"- 几何平均: {metrics['g_mean']:.4f}")
    #
    #     print("\nAUC指标:")
    #     print(f"- ROC AUC: {metrics['roc_auc']:.4f}")
    #     print(f"- PR AUC: {metrics['pr_auc']:.4f}")
    #
    #     print("\n混淆矩阵:")
    #     print(f"真阴性 (TN): {tn}")
    #     print(f"假阳性 (FP): {fp}")
    #     print(f"假阴性 (FN): {fn}")
    #     print(f"真阳性 (TP): {tp}")
    #
    #     # 7. 计算额外的统计信息
    #     total = tn + fp + fn + tp
    #     anomaly_ratio = (tp + fn) / total if total > 0 else 0
    #
    #     print(f"\n检测统计:")
    #     print(f"- 总边数: {total}")
    #     print(f"- 检测为异常: {tp + fp}")
    #     print(f"- 真实异常: {tp + fn}")
    #     print(f"- 异常比例: {anomaly_ratio * 100:.2f}%")
    #
    #     # 8. 分析预测分布
    #     print("\n分数分布:")
    #     print(results.groupby('predicted')['score'].describe())
    #
    #     return metrics

    # 2
    # def evaluate_detection(self, results, threshold):
    #     """
    #     优化的边异常检测评估函数
    #     时间: 2025-02-11 05:17:56
    #     用户: 123wzj
    #     """
    #     print("\n评估检测结果...")
    #
    #     # 1. 数据验证
    #     if results.empty:
    #         raise ValueError("结果数据为空")
    #
    #     # 2. 阈值分析和优化
    #     print("\n分数分布分析:")
    #     quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    #     score_distribution = results['score'].quantile(quantiles)
    #     print("分数分位数分布:")
    #     for q, score in zip(quantiles, score_distribution):
    #         print(f"{q * 100}%分位数: {score:.4f}")
    #
    #     # 2.1 根据分布确定最优阈值
    #     mean_score = results['score'].mean()
    #     std_score = results['score'].std()
    #
    #     # 2.2 计算候选阈值
    #     candidate_thresholds = {
    #         '原始阈值': threshold,
    #         '均值+标准差': mean_score + std_score,
    #         '均值+2倍标准差': mean_score + 2 * std_score,
    #         '90%分位数': results['score'].quantile(0.90),
    #         '95%分位数': results['score'].quantile(0.95)
    #     }
    #
    #     # 2.3 评估各个阈值的性能
    #     best_threshold = threshold
    #     best_f1 = 0
    #
    #     print("\n阈值评估:")
    #     for threshold_name, thresh in candidate_thresholds.items():
    #         # 临时预测
    #         temp_pred = (results['score'] > thresh).astype(int)
    #         # 使用95%分位数作为真实标签
    #         true_anomalies = (results['score'] > results['score'].quantile(0.95)).astype(int)
    #
    #         # 计算指标
    #         tp = sum((true_anomalies == 1) & (temp_pred == 1))
    #         fp = sum((true_anomalies == 0) & (temp_pred == 1))
    #         fn = sum((true_anomalies == 1) & (temp_pred == 0))
    #
    #         # 计算精确率和召回率
    #         prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    #         rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    #         f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    #
    #         print(f"\n{threshold_name}:")
    #         print(f"- 阈值: {thresh:.4f}")
    #         print(f"- 精确率: {prec:.4f}")
    #         print(f"- 召回率: {rec:.4f}")
    #         print(f"- F1分数: {f1:.4f}")
    #
    #         # 更新最佳阈值
    #         if f1 > best_f1:
    #             best_f1 = f1
    #             best_threshold = thresh
    #
    #     print(f"\n选择最佳阈值: {best_threshold:.4f}")
    #
    #     # 3. 使用最优阈值进行预测
    #     predictions = (results['score'] > best_threshold).astype(int)
    #     results['predicted'] = predictions
    #
    #     # 4. 设置真实标签
    #     anomaly_threshold = results['score'].quantile(0.95)
    #     results['is_anomaly'] = (results['score'] > anomaly_threshold).astype(int)
    #
    #     # 5. 计算混淆矩阵
    #     tn = sum((results['is_anomaly'] == 0) & (predictions == 0))
    #     fp = sum((results['is_anomaly'] == 0) & (predictions == 1))
    #     fn = sum((results['is_anomaly'] == 1) & (predictions == 0))
    #     tp = sum((results['is_anomaly'] == 1) & (predictions == 1))
    #
    #     # 6. 计算评估指标
    #     metrics = {}
    #     metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    #     metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    #     metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    #
    #     if metrics['precision'] + metrics['recall'] > 0:
    #         metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
    #                         (metrics['precision'] + metrics['recall'])
    #     else:
    #         metrics['f1'] = 0
    #
    #     # 其他指标保持不变
    #     metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    #     metrics['g_mean'] = np.sqrt(metrics['recall'] * metrics['specificity'])
    #
    #     try:
    #         metrics['roc_auc'] = roc_auc_score(results['is_anomaly'], results['score'])
    #         precision_curve, recall_curve, _ = precision_recall_curve(
    #             results['is_anomaly'],
    #             results['score']
    #         )
    #         metrics['pr_auc'] = auc(recall_curve, precision_curve)
    #     except:
    #         metrics['roc_auc'] = metrics['pr_auc'] = 0
    #
    #     metrics['confusion_matrix'] = {
    #         'tn': int(tn),
    #         'fp': int(fp),
    #         'fn': int(fn),
    #         'tp': int(tp)
    #     }
    #
    #     # 7. 输出详细评估结果
    #     print(f"\n最终评估结果:")
    #     print(f"使用最优阈值: {best_threshold:.4f}")
    #     print("\n基础指标:")
    #     print(f"- 精确率: {metrics['precision']:.4f}")
    #     print(f"- 召回率: {metrics['recall']:.4f}")
    #     print(f"- 特异度: {metrics['specificity']:.4f}")
    #     print(f"- F1分数: {metrics['f1']:.4f}")
    #
    #     # 8. 异常检测统计
    #     total = tn + fp + fn + tp
    #     anomaly_ratio = (tp + fn) / total if total > 0 else 0
    #
    #     print(f"\n检测统计:")
    #     print(f"- 总边数: {total}")
    #     print(f"- 检测为异常: {tp + fp}")
    #     print(f"- 真实异常: {tp + fn}")
    #     print(f"- 异常比例: {anomaly_ratio * 100:.2f}%")
    #
    #     # 9. 分数分布分析
    #     print("\n分数分布:")
    #     print(results.groupby('predicted')['score'].describe())
    #
    #     return metrics

    def evaluate_detection(self, results, threshold):
        """
        评估边异常检测结果，使用实际数据和合理的阈值策略
        Time: 2025-02-11 05:28:05
        User: 123wzj
        """
        print("\n评估检测结果...")

        # 1. 数据验证
        if results.empty:
            raise ValueError("结果数据为空")

        # 2. 分析实际数据分布
        print("\n=== 数据分布分析 ===")
        score_stats = results['score'].describe()
        print(score_stats)

        # 3. 计算预测结果（使用输入的阈值）
        predictions = (results['score'] > threshold).astype(int)
        results['predicted'] = predictions

        # 4. 使用实际数据的95%分位数作为异常标签阈值
        anomaly_threshold = results['score'].quantile(0.95)
        results['is_anomaly'] = (results['score'] > anomaly_threshold).astype(int)

        # 5. 计算混淆矩阵
        tn = sum((results['is_anomaly'] == 0) & (predictions == 0))
        fp = sum((results['is_anomaly'] == 0) & (predictions == 1))
        fn = sum((results['is_anomaly'] == 1) & (predictions == 0))
        tp = sum((results['is_anomaly'] == 1) & (predictions == 1))

        # 6. 计算评估指标
        metrics = {}
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                            (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0

        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        metrics['g_mean'] = np.sqrt(metrics['recall'] * metrics['specificity'])

        # ROC和PR曲线相关指标
        try:
            metrics['roc_auc'] = roc_auc_score(results['is_anomaly'], results['score'])
            precision_curve, recall_curve, _ = precision_recall_curve(
                results['is_anomaly'],
                results['score']
            )
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
        except:
            metrics['roc_auc'] = 0
            metrics['pr_auc'] = 0

        metrics['confusion_matrix'] = {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }

        # 7. 输出评估结果
        print("\n=== 评估结果 ===")
        print(f"使用阈值: {threshold:.4f}")

        print("\n基础指标:")
        print(f"- 精确率: {metrics['precision']:.4f}")
        print(f"- 召回率: {metrics['recall']:.4f}")
        print(f"- 特异度: {metrics['specificity']:.4f}")
        print(f"- F1分数: {metrics['f1']:.4f}")

        print("\n平衡指标:")
        print(f"- 平衡准确率: {metrics['balanced_accuracy']:.4f}")
        print(f"- 几何平均: {metrics['g_mean']:.4f}")

        print("\n混淆矩阵:")
        print(f"真阴性 (TN): {tn}")
        print(f"假阳性 (FP): {fp}")
        print(f"假阴性 (FN): {fn}")
        print(f"真阳性 (TP): {tp}")

        # 8. 实际数据统计
        total = tn + fp + fn + tp
        anomaly_ratio = (tp + fn) / total if total > 0 else 0

        print("\n检测统计:")
        print(f"- 总边数: {total}")
        print(f"- 检测为异常: {tp + fp}")
        print(f"- 真实异常: {tp + fn}")
        print(f"- 异常比例: {anomaly_ratio * 100:.2f}%")

        # 9. 分析实际预测分布
        print("\n分数分布:")
        print(results.groupby('predicted')['score'].describe())

        # 10. 阈值建议（基于实际数据）
        if metrics['precision'] < 0.9:
            print("\n建议: 当前精确率较低，考虑提高阈值")
            higher_threshold = results['score'].quantile(0.97)
            print(f"建议阈值: {higher_threshold:.4f}")
        elif metrics['recall'] < 0.5:
            print("\n建议: 当前召回率较低，考虑适当降低阈值")
            lower_threshold = results['score'].quantile(0.93)
            print(f"建议阈值: {lower_threshold:.4f}")

        return metrics


    # 完整的训练和评估流程
    def train_and_evaluate(self, train_graphs, val_graphs, test_graphs, node_list):
        # 1. 训练短期模型
        print("训练短期模型...")
        short_term_embeddings = self.train_short_term_model(train_graphs, node_list)

        # 2. 训练长期模型
        print("训练长期模型...")
        long_term_embeddings = self.train_long_term_model(short_term_embeddings)
        # 保存嵌入
        embeddings = {
            'short_term': short_term_embeddings,
            'long_term': long_term_embeddings
        }
        # 3. 验证阶段
        print("验证模型...")
        val_metrics, optimal_threshold = self.validate_model(val_graphs, embeddings)
        print(f"验证指标: {val_metrics}")

        # 4. 测试阶段
        print("测试模型...")
        test_metrics = self.test_model(test_graphs, embeddings, optimal_threshold)
        print(f"测试指标: {test_metrics}")

        # 5.保存结果
        self.save_results(embeddings, val_metrics, test_metrics, optimal_threshold)
        return embeddings, val_metrics, test_metrics

    # 保存模型
    def save_results(self, embeddings, val_metrics, test_metrics, threshold):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = Path(output_folder) / 'results'
        results_path.mkdir(parents=True, exist_ok=True)
        # with open(f"{results_path}/short_term_embeddings_{timestamp}.pkl", 'wb') as f:
        #     pickle.dump(embeddings['short_term'], f)
        #
        # with open(f"{results_path}/long_term_embeddings_{timestamp}.pkl", 'wb') as f:
        #     pickle.dump(embeddings['long_term'], f)

        with open(f"{results_path}/validation_metrics_{timestamp}.json", 'w') as f:
            json.dump(val_metrics, f, indent=4)

        with open(f"{results_path}/test_metrics_{timestamp}.json", 'w') as f:
            json.dump(test_metrics, f, indent=4)

        with open(f"{results_path}/optimal_threshold_{timestamp}.txt", 'w') as f:
            f.write(str(threshold))

        print(f"结果已保存到 {results_path}")


# 加载已保存的 node_map 和 graphs 数据
def load_saved_data(data_folder,graphs_file):
    # 加载节点映射
    node_map_path = f"{data_folder}/lanl/node_map_lanl.pickle"
    print(f"正在加载节点映射从: {node_map_path}")
    with open(node_map_path, 'rb') as f:
        node_map = pickle.load(f)
    # print(f"成功加载节点映射，包含 {len(node_map)} 个节点")
    # 加载图数据
    print(f"正在加载图数据从: {graphs_file}")
    with open(graphs_file, 'rb') as f:
        graphs = pickle.load(f)
    return node_map, graphs

def main():
    # 解析参数
    args = parse_args()
    # 加载数据  图快照 125
    node_map, graphs = load_saved_data(data_folder,graphs_file)

    # 获取总节点数  208207
    # print(f"node_map大小: {len(node_map)}")

    if node_map is None or graphs is None:
        print("Failed to load data")
        return

    # 使用验证集的情况
    # train_range, val_range, test_range = get_data_snapshots_split(graphs_file, use_validation=True)
    # 不使用验证集的索引
    train_range, val_range, test_range = get_data_snapshots_split(graphs_file, use_validation=False)
    # 获取训练集图快照
    train_graphs = graphs[train_range[0]:train_range[1]]
    # 获取验证集图快照
    val_graphs = graphs[val_range[0]:val_range[1]]
    # 获取测试集图快照
    test_graphs = graphs[test_range[0]:test_range[1]]

    # 获取节点列表
    node_list = list(node_map.values())

    # 初始化训练管理器
    trainer = TrainingManager(args, node_list, node_map, graphs)

    # 检查是否存在预训练模型
    # model_path = os.path.join(output_folder, 'model', 'pretrained_model.pth')
    # embeddings_path = os.path.join(output_folder, 'embeddings', 'pretrained_embeddings.pkl')


    # 训练并评估
    embeddings, val_metrics, test_metrics = trainer.train_and_evaluate(
        train_graphs, val_graphs, test_graphs, node_list
    )
    # 打印最终结果
    print("\n最终结果：")
    print("\n验证集指标：")
    print(f"ROC AUC: {val_metrics['roc_auc']:.4f}")
    print(f"PR AUC: {val_metrics['pr_auc']:.4f}")
    print(f"精确率: {val_metrics['precision']:.4f}")
    print(f"召回率: {val_metrics['recall']:.4f}")

    print("\n测试集指标：")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"PR AUC: {test_metrics['pr_auc']:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"召回率: {test_metrics['recall']:.4f}")
    memory_cleanup()

if __name__ == "__main__":
    main()