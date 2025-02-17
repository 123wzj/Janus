import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalAttentionLayer(torch.nn.Module):
    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
                 n_head=2, dropout=0.1, output_dimension=None):
        super(TemporalAttentionLayer, self).__init__()

        self.n_head = n_head
        if output_dimension is None:
            self.output_dimension = n_node_features
        else:
            self.output_dimension = output_dimension

        # 特征转换层
        self.node_transform = nn.Linear(n_node_features, self.output_dimension)
        self.neighbor_transform = nn.Linear(n_neighbors_features, self.output_dimension)
        self.edge_transform = nn.Linear(n_edge_features, self.output_dimension)
        self.time_transform = nn.Linear(time_dim, self.output_dimension)

        # 序列长度统一转换层
        self.sequence_transform = nn.Sequential(
            nn.Linear(self.output_dimension, self.output_dimension),
            nn.ReLU(),
            nn.Linear(self.output_dimension, self.output_dimension)
        )

        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.output_dimension,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(self.output_dimension)
        self.layer_norm2 = nn.LayerNorm(self.output_dimension)

    def forward(self, src_node_features, src_time_features, neighbor_features,
                edge_time_features, edge_features, mask=None):
        # 1. 首先确定统一的批次大小（使用最小的批次大小）
        batch_sizes = [
            src_node_features.size(0),
            neighbor_features.size(0),
            edge_features.size(0),
            edge_time_features.size(0)
        ]
        batch_size = min(batch_sizes)  # 使用最小的批次大小

        # 2. 裁剪所有输入到相同的批次大小
        src_node_features = src_node_features[:batch_size]
        if src_time_features.dim() == 3:
            src_time_features = src_time_features[:batch_size]
        else:
            src_time_features = src_time_features[:batch_size]
        neighbor_features = neighbor_features[:batch_size]
        edge_features = edge_features[:batch_size]
        edge_time_features = edge_time_features[:batch_size]
        if mask is not None:
            mask = mask[:batch_size]

        # 3. 特征转换
        node_h = self.node_transform(src_node_features)  # [B, D]

        # 处理时间特征
        if src_time_features.dim() == 3:
            time_h = self.time_transform(src_time_features.mean(dim=1))  # [B, D]
        else:
            time_h = self.time_transform(src_time_features)  # [B, D]

        # 4. 转换其他特征
        neighbor_h = self.neighbor_transform(neighbor_features)  # [B, S1, D]
        edge_h = self.edge_transform(edge_features)  # [B, S2, D]
        edge_time_h = self.time_transform(edge_time_features)  # [B, S3, D]

        # 5. 找到序列的最小长度
        seq_lens = [
            neighbor_h.size(1),
            edge_h.size(1),
            edge_time_h.size(1)
        ]
        min_seq_len = min(seq_lens)

        # 6. 裁剪序列到最小长度
        neighbor_h = neighbor_h[:, :min_seq_len]
        edge_h = edge_h[:, :min_seq_len]
        edge_time_h = edge_time_h[:, :min_seq_len]

        # 7. 特征融合
        combined_features = (neighbor_h + edge_h + edge_time_h) / 3.0

        # 8. 准备注意力输入
        query = node_h.unsqueeze(1)  # [B, 1, D]
        key = combined_features  # [B, S, D]
        value = combined_features

        # 9. 处理掩码
        if mask is not None:
            mask = mask[:, :min_seq_len]
            attention_mask = mask.bool()
        else:
            attention_mask = None

        # 10. 多头注意力
        attn_output, attn_weights = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=attention_mask
        )

        # 11. 残差连接和层归一化
        output = self.layer_norm1(query + attn_output)
        output = output.squeeze(1)  # [B, D]

        # 12. 最终输出
        output = self.layer_norm2(output + node_h)
        output = self.dropout(output)

        return output, attn_weights