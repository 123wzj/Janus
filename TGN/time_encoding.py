import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()
    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    # 初始化权重
    self.w.weight = torch.nn.Parameter(
      (torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
      .float().reshape(dimension, -1)
    )
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    """
    时间编码前向传播
    Args:
        t: 时间戳张量
    """
    # 确保输入是2D张量
    if t.dim() == 1:
      t = t.unsqueeze(0)  # [seq_len] -> [1, seq_len]

    # 添加最后一个维度用于线性层
    t = t.unsqueeze(-1)  # [batch_size, seq_len] -> [batch_size, seq_len, 1]

    # 应用余弦变换
    output = torch.cos(self.w(t))  # [batch_size, seq_len, dimension]

    return output
