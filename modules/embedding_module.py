import torch
from torch import nn
import numpy as np
import math

from TGN.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20,
                          time_diffs=None, use_time_proj=True):
        pass


class DimensionConverter:
    """维度转换工具类"""

    @staticmethod
    def align_dimensions(source_array, target_length, method='interpolate'):
        """
        将源数组的维度对齐到目标长度

        参数:
            source_array: 源数组
            target_length: 目标长度
            method: 转换方法 ['interpolate', 'repeat', 'pad']
        """
        source_array = np.asarray(source_array)
        source_length = len(source_array)

        if source_length == target_length:
            return source_array

        if method == 'interpolate':
            # 使用线性插值
            indices = np.linspace(0, source_length - 1, target_length)
            return np.interp(indices, np.arange(source_length), source_array)

        elif method == 'repeat':
            # 重复数据
            repeat_times = int(np.ceil(target_length / source_length))
            repeated = np.tile(source_array, repeat_times)
            return repeated[:target_length]

        elif method == 'pad':
            # 填充数据
            result = np.zeros(target_length)
            if source_length < target_length:
                result[:source_length] = source_array
                result[source_length:] = source_array[-1]
            else:
                result = source_array[:target_length]
            return result

        raise ValueError(f"Unsupported method: {method}")


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, dropout)

        self.use_memory = use_memory
        self.device = device
        self.embedding_dimension = embedding_dimension
        self.dimension_converter = DimensionConverter()

        # 特征转换层
        self.node_dim_transform = nn.Linear(n_node_features, embedding_dimension)
        self.time_dim_transform = nn.Linear(n_time_features, embedding_dimension)
        self.edge_dim_transform = nn.Linear(n_edge_features, embedding_dimension)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20,
                          time_diffs=None, use_time_proj=True):
        """计算节点嵌入"""
        # 1. 输入预处理
        source_nodes = np.asarray(source_nodes)
        timestamps = np.asarray(timestamps)

        # 2. 获取维度信息
        source_length = len(source_nodes)
        timestamp_length = len(timestamps)

        # 3. 对齐维度
        if source_length != timestamp_length:
            if source_length > timestamp_length:
                # 扩展时间戳到源节点长度
                timestamps = self.dimension_converter.align_dimensions(
                    timestamps, source_length, method='interpolate')
            else:
                # 扩展源节点到时间戳长度
                source_nodes = self.dimension_converter.align_dimensions(
                    source_nodes, timestamp_length, method='repeat')

        # 4. 获取有效节点
        max_node_idx = self.node_features.size(0) - 1
        valid_mask = source_nodes <= max_node_idx

        # 5. 确保维度匹配
        assert len(valid_mask) == len(timestamps), \
            f"Dimension mismatch: valid_mask={len(valid_mask)}, timestamps={len(timestamps)}"

        # 6. 过滤有效数据
        valid_source_nodes = source_nodes[valid_mask]
        valid_timestamps = timestamps[valid_mask]

        if len(valid_source_nodes) == 0:
            return torch.zeros((len(source_nodes), self.embedding_dimension),
                               device=self.device)

        source_nodes_torch = torch.from_numpy(valid_source_nodes).long().to(self.device)
        timestamps_torch = torch.from_numpy(valid_timestamps).float().to(self.device)

        source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))
        source_nodes_time_embedding = self.time_dim_transform(source_nodes_time_embedding)

        source_node_features = self.node_features[source_nodes_torch, :]
        source_node_features = self.node_dim_transform(source_node_features)

        if n_layers == 0:
            full_features = torch.zeros((len(source_nodes), self.embedding_dimension),
                                        device=self.device)
            full_features[valid_mask] = source_node_features
            return full_features

        # 7.获取邻居信息
        neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
            valid_source_nodes,
            valid_timestamps,
            n_neighbors=n_neighbors)

        # 8. 转换特征
        neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
        edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

        # 9. 计算时间差
        valid_timestamps = valid_timestamps.reshape(-1, 1)
        edge_times = edge_times.reshape(valid_timestamps.shape[0], -1)
        edge_deltas = valid_timestamps - edge_times
        edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

        # 10. 递归计算邻居嵌入
        neighbors = neighbors.flatten()
        neighbor_embeddings = self.compute_embedding(
            memory,
            neighbors,
            np.repeat(valid_timestamps.flatten(), n_neighbors),
            n_layers=n_layers - 1,
            n_neighbors=n_neighbors)

        # 11. 处理特征维度
        neighbor_embeddings = neighbor_embeddings.view(len(valid_source_nodes), n_neighbors, -1)

        edge_time_embeddings = self.time_encoder(edge_deltas_torch)
        edge_time_embeddings = self.time_dim_transform(edge_time_embeddings)
        edge_time_embeddings = edge_time_embeddings.view(len(valid_source_nodes), n_neighbors, -1)

        edge_features = self.edge_features[edge_idxs, :]
        edge_features = edge_features.view(len(valid_source_nodes), n_neighbors, -1)
        edge_features = self.edge_dim_transform(edge_features)

        # 12. 创建mask并聚合
        mask = neighbors_torch == 0

        valid_embedding = self.aggregate(
            n_layers,
            source_node_features,
            source_nodes_time_embedding,
            neighbor_embeddings,
            edge_time_embeddings,
            edge_features,
            mask)

        # 13. 创建完整输出
        full_embedding = torch.zeros((len(source_nodes), self.embedding_dimension),
                                     device=self.device)
        full_embedding[valid_mask] = valid_embedding

        return full_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory)

        # 修改注意力层初始化
        self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
            n_node_features=embedding_dimension,
            n_neighbors_features=embedding_dimension,
            n_edge_features=embedding_dimension,
            time_dim=embedding_dimension,
            n_head=n_heads,
            dropout=dropout,
            output_dimension=embedding_dimension)
            for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        """
        聚合节点特征

        参数:
            source_node_features: [B, D]
            source_nodes_time_embedding: [B, D]
            neighbor_embeddings: [B, S, D]
            edge_time_embeddings: [B, S, D]
            edge_features: [B, S, D]
            mask: [B, S]
        """
        attention_model = self.attention_models[n_layer - 1]

        # 获取维度信息
        batch_size = source_node_features.size(0)
        n_neighbors = neighbor_embeddings.size(1)

        # 确保维度一致性
        source_node_features = source_node_features.view(batch_size, -1)
        source_nodes_time_embedding = source_nodes_time_embedding.view(batch_size, -1)
        neighbor_embeddings = neighbor_embeddings.view(batch_size, n_neighbors, -1)
        edge_time_embeddings = edge_time_embeddings.view(batch_size, n_neighbors, -1)
        edge_features = edge_features.view(batch_size, n_neighbors, -1)

        if mask is not None:
            mask = mask.view(batch_size, n_neighbors)

        # 应用注意力机制
        source_embedding, _ = attention_model(
            source_node_features,
            source_nodes_time_embedding,
            neighbor_embeddings,
            edge_time_embeddings,
            edge_features,
            mask)

        return source_embedding


# 保持其他类的实现不变
class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                edge_features=edge_features,
                                                memory=memory,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder,
                                                n_layers=n_layers,
                                                n_node_features=n_node_features,
                                                n_edge_features=n_edge_features,
                                                n_time_features=n_time_features,
                                                embedding_dimension=embedding_dimension,
                                                device=device,
                                                n_heads=n_heads,
                                                dropout=dropout,
                                                use_memory=use_memory)

        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                             n_edge_features, embedding_dimension)
                                             for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                             embedding_dimension) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                       dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory,
            neighbor_finder=neighbor_finder,
            time_encoder=time_encoder,
            n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features,
            embedding_dimension=embedding_dimension,
            device=device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory)
    else:
        raise ValueError(f"Embedding Module {module_type} not supported")