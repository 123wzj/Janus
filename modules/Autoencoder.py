from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class AutoencoderModel(nn.Module):
    def __init__(self, mask_val):
        super(AutoencoderModel, self).__init__()
        self.mask_val = mask_val

    def autoencoder_model(self, time_step, dim):
        # 使用较小的隐藏层大小，但保持足够的表达能力
        hidden_size = min(32, dim)  # 动态设置隐藏层大小

        # 使用单层GRU来减少参数量
        self.encoder = nn.GRU(dim, hidden_size, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, dim)

        # 添加批归一化以提高训练稳定性
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        return self

    def forward(self, x):
        # 使用稀疏掩码
        mask = (x != self.mask_val).float()
        x = x * mask

        # 编码
        encoded, _ = self.encoder(x)
        encoded = self.batch_norm(encoded.transpose(1, 2)).transpose(1, 2)

        # 解码
        decoded, _ = self.decoder(encoded)
        output = self.output_layer(decoded)

        return output, encoded


def long_term_embedding(self, short_term_embs):
    # 转换为numpy数组以节省内存
    if isinstance(short_term_embs, torch.Tensor):
        short_term_embs = short_term_embs.numpy()

    total_node, time_step, dim = short_term_embs.shape

    # 创建内存效率更高的数据集
    class EfficientDataset(torch.utils.data.Dataset):
        def __init__(self, data, chunk_size=1000):
            self.data = data
            self.chunk_size = chunk_size
            self.num_chunks = int(np.ceil(len(data) / chunk_size))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            chunk_idx = idx // self.chunk_size
            if not hasattr(self, 'current_chunk') or self.current_chunk_idx != chunk_idx:
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, len(self.data))
                self.current_chunk = torch.tensor(
                    self.data[start_idx:end_idx],
                    dtype=torch.float32
                )
                self.current_chunk_idx = chunk_idx

            local_idx = idx % self.chunk_size
            return self.current_chunk[local_idx]

    # 创建数据加载器
    dataset = EfficientDataset(short_term_embs)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,  # 适中的批次大小
        shuffle=False,
        num_workers=0,  # 单进程加载以避免内存问题
        pin_memory=True
    )

    # 初始化模型
    self.model = AutoencoderModel(mask_val=0.).autoencoder_model(time_step, dim)
    self.model = self.model.to(self.device)

    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # 训练循环
    best_loss = float('inf')
    loss_history = []
    patience = 0
    max_patience = 10

    print("\n开始训练...")
    for epoch in range(self.args.epoch):
        self.model.train()
        epoch_loss = 0
        batch_count = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output, _ = self.model(batch)
                    loss = criterion(output, batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output, _ = self.model(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # 定期清理内存
            if batch_count % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{self.args.epoch}, Loss: {avg_loss:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 早停
        if patience >= max_patience:
            print(f"早停触发 - {patience} 个epoch没有改善")
            break

    # 保存训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('fig/train_error.png')
    plt.close()

    # 生成最终嵌入
    print("\n生成最终嵌入...")
    self.model.eval()
    embeddings = []

    with torch.no_grad():
        for idx in range(0, total_node, 100):  # 每次处理100个节点
            batch = torch.tensor(
                short_term_embs[idx:idx + 100],
                dtype=torch.float32
            ).to(self.device)
            output, _ = self.model(batch)
            embeddings.append(output.cpu().numpy())

            if idx % 1000 == 0:
                print(f"处理进度: {idx}/{total_node}")

    embeddings = np.concatenate(embeddings, axis=0)
    print(f"完成! 最终嵌入形状: {embeddings.shape}")

    return embeddings.transpose(1, 0, 2)
