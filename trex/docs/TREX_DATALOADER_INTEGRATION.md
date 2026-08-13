# T-Rex VQ-VAE 与你的 Dataloader 集成完成

## ✅ 集成概览

已成功将T-Rex官方VQ-VAE与你的dataloader (`/home/yang/TactileEncoder/dataloader`) 完全集成。

---

## 📊 你的数据格式

### 输入数据结构
```python
batch = {
    'tactile_history': torch.Tensor,  # [B, 16, 12]
    # 其他字段...
}
```

### 数据组织
- **形状**: `[B, T, 12]` 其中 T=16 (历史窗口长度)
- **通道顺序**: 
  - `[..., 0:6]` = Finger 0 的 6D 力/力矩 `[Fx, Fy, Fz, Mx, My, Mz]`
  - `[..., 6:12]` = Finger 1 的 6D 力/力矩 `[Fx, Fy, Fz, Mx, My, Mz]`

### 数据源
- **数据集**: `/home/yang/TactileEncoder/dataset/so101/dm_350_400_merged`
- **配置文件**: `dataloader/vqvae_tactile.yaml`
- **训练样本**: 150,908
- **验证样本**: 15,471
- **分割比例**: 90% train / 10% val

---

## 🔧 已完成的适配

### 1. 统计量计算 (`compute_trex_stats.py`)
```bash
python TactileSelfencoder/compute_trex_stats.py \
    --data_config dataloader/vqvae_tactile.yaml \
    --output TactileSelfencoder/trex_tactile_stats.json
```

**输出统计**: `trex_tactile_stats.json`
```json
{
  "mean": [[...], [...]],     // [2, 6] per-finger per-channel
  "std": [[...], [...]],       // [2, 6]
  "min": [[...], [...]],
  "max": [[...], [...]],
  "n_samples": 2414528,        // 150908 samples × 16 frames
  "n_fingers": 2,
  "force_dim": 6
}
```

### 2. 数据预处理
自动应用：
- **Reshape**: `[B, 16, 12]` → `[B, 16, 2, 6]`
- **Per-finger normalization**: 使用训练集统计量
- **批处理**: 支持你的dataloader的所有配置

### 3. 训练脚本 (`train_trex_vqvae.py`)
完全兼容你的dataloader：
```python
from dataloader.dataloader import build_base_dataset, build_dataloaders

# 直接使用你的dataloader
dataset = build_base_dataset(data_config)
train_loader, val_loader = build_dataloaders(
    dataset, 
    batch_size, 
    num_workers
)

# 自动处理batch格式
for batch in train_loader:
    force_raw = batch['tactile_history']  # [B, 16, 12]
    # 自动转换为 [B, 16, 2, 6]
```

### 4. 推理脚本 (`inference_trex.py`)
支持：
- 从验证集加载数据
- 自动应用训练时的标准化
- 生成重建可视化
- 导出编码结果

---

## 🚀 快速开始

### 方法1: 一键运行完整流程
```bash
./run_trex_vqvae.sh
```

这会自动执行：
1. 计算统计量（如果不存在）
2. 训练模型（30 epochs）
3. 生成可视化

### 方法2: 分步执行

#### Step 1: 计算统计量
```bash
python TactileSelfencoder/compute_trex_stats.py \
    --data_config dataloader/vqvae_tactile.yaml \
    --output TactileSelfencoder/trex_tactile_stats.json
```

#### Step 2: 训练
```bash
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/trex_vqvae
```

#### Step 3: 推理和可视化
```bash
python TactileSelfencoder/inference_trex.py \
    --checkpoint outputs/trex_vqvae/latest.pt \
    --data_config dataloader/vqvae_tactile.yaml \
    --output outputs/trex_vqvae/visualizations
```

### 方法3: 快速测试（3 epochs）
```bash
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_test_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/trex_test
```

---

## 📝 配置文件

### 训练配置: `trex_2finger_config.yaml`
```yaml
model:
  granularity: finger      # per-finger 量化
  n_fingers: 2             # 你的系统使用2根手指
  per_finger_dim: 6        # 6D 力/力矩
  window: 16               # 历史窗口长度
  codebook_size: 64        # 码本大小
  embed_dim: 256           # 嵌入维度

train:
  epochs: 30
  batch_size: 128
  lr: 3.0e-4
```

### 数据配置: `dataloader/vqvae_tactile.yaml`
```yaml
dataset:
  repo_id: /home/yang/TactileEncoder/dataset/so101/dm_350_400_merged
  keys:
    tactile_type: vqvae
    tactile_force:
      - observation.tactile.right_force
      - observation.tactile.left_force

sequence:
  tactile_history_force: 16  # 与T-Rex要求一致
  length: 16

split:
  train: 0.9
  val: 0.1
  seed: 42

loader:
  batch_size: 16
  num_workers: 8
```

---

## 🔍 数据流详解

### 1. Dataloader输出
```python
batch = next(iter(train_loader))
batch['tactile_history'].shape  # [16, 16, 12]
# B=16, T=16, Channels=12
```

### 2. Reshape操作
```python
force_raw = batch['tactile_history']  # [B, 16, 12]
force_raw = force_raw.reshape(B, 16, 2, 6)  # [B, 16, 2, 6]
```

### 3. 标准化
```python
# 使用训练集统计量
force_norm = (force_raw - mean) / (std + 1e-6)
# mean.shape = [2, 6]
# std.shape = [2, 6]
```

### 4. 模型输入
```python
# Permute to [B, 2, 16, 6]
x = force_norm.permute(0, 2, 1, 3)

# Forward pass
z_q, indices, vq_loss, perplexity = model(x)
# indices.shape = [B, 2]  # 每根手指一个token
```

### 5. 输出解释
```python
indices[0, 0]  # Finger 0 的码字索引 (0-63)
indices[0, 1]  # Finger 1 的码字索引 (0-63)
```

---

## 📈 预期训练指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 重建损失 (MSE) | < 0.02 | 标准化后的均方误差 |
| VQ损失 | < 0.5 | 量化损失 |
| 码本利用率 | > 90% | 活跃码字比例 |
| 困惑度 (Perplexity) | 50-60 | 码字使用均匀度 |

### 训练曲线示例
```
Epoch 1/30:
  Step   50: recon_loss=0.0856, vq_loss=0.2341, perplexity=32.4, usage=45.3%
  Step  100: recon_loss=0.0421, vq_loss=0.1892, perplexity=41.2, usage=67.2%
  ...
Epoch 10/30:
  Step 5000: recon_loss=0.0134, vq_loss=0.0823, perplexity=56.8, usage=93.8%
  ...
```

---

## 🗂️ 输出文件结构

```
outputs/trex_vqvae/
├── latest.pt              # 最新checkpoint
├── best.pt                # 最佳checkpoint (最低验证损失)
├── epoch_5.pt             # 每5个epoch保存一次
├── epoch_10.pt
├── ...
├── training.log           # 训练日志
└── visualizations/        # 推理输出
    ├── reconstruction_stats.json
    ├── sample_000.png
    ├── sample_001.png
    └── ...
```

### Checkpoint内容
```python
checkpoint = torch.load('outputs/trex_vqvae/best.pt')
checkpoint.keys()
# ['model_state_dict', 'optimizer_state_dict', 'epoch', 'step',
#  'best_val_loss', 'train_loss', 'val_loss', 'config', 'stats']
```

---

## 🔬 验证和调试

### 1. 检查数据加载
```python
python TactileSelfencoder/compute_trex_stats.py \
    --data_config dataloader/vqvae_tactile.yaml \
    --output test_stats.json

# 检查统计量
cat test_stats.json | jq '.mean, .std'
```

### 2. 快速训练测试
```bash
# 只训练3个epoch
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_test_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/test
```

### 3. 验证模型前向传播
```python
from TactileSelfencoder.trex_official.tactile_vqvae import TactileVQVAE

model = TactileVQVAE(config).cuda()
x = torch.randn(4, 2, 16, 6).cuda()  # [B, n_fingers, T, 6]

z_q, indices, vq_loss, perplexity = model(x)
print(f"Indices shape: {indices.shape}")  # [4, 2]
print(f"VQ loss: {vq_loss.item():.4f}")
print(f"Perplexity: {perplexity.item():.2f}")
```

---

## 🐛 常见问题

### Q1: 为什么需要重新计算统计量？
**A**: 训练集统计量用于标准化。必须只使用训练集（不包括验证集）来避免数据泄漏。

### Q2: 如何修改batch size？
**A**: 有两个地方可以设置：
- `trex_2finger_config.yaml` 中的 `train.batch_size` (模型训练)
- `dataloader/vqvae_tactile.yaml` 中的 `loader.batch_size` (数据加载)

建议两者保持一致。

### Q3: 训练多久合适？
**A**: 
- 快速测试: 3 epochs (~10分钟)
- 正常训练: 30 epochs (~2小时)
- 完整训练: 100 epochs (~6小时)

### Q4: 如何恢复训练？
**A**: 训练脚本会自动检测并加载 `latest.pt`：
```bash
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/trex_vqvae  # 相同目录
```

### Q5: 如何调整码本大小？
**A**: 修改 `trex_2finger_config.yaml`:
```yaml
model:
  codebook_size: 128  # 从64增加到128
```

---

## 📚 文件索引

### 核心代码
- `TactileSelfencoder/trex_official/tactile_vqvae.py` - T-Rex官方VQ-VAE模型
- `TactileSelfencoder/trex_official/f6_encoder.py` - Per-finger编码器
- `TactileSelfencoder/trex_official/f6_decoder.py` - Per-finger解码器
- `TactileSelfencoder/trex_official/vqvae_ema.py` - EMA量化器

### 训练和推理
- `TactileSelfencoder/train_trex_vqvae.py` - 训练脚本
- `TactileSelfencoder/compute_trex_stats.py` - 统计量计算
- `TactileSelfencoder/inference_trex.py` - 推理和可视化

### 配置文件
- `TactileSelfencoder/trex_2finger_config.yaml` - 完整训练配置
- `TactileSelfencoder/trex_2finger_test_config.yaml` - 快速测试配置
- `dataloader/vqvae_tactile.yaml` - 你的数据配置

### 数据
- `TactileSelfencoder/trex_tactile_stats.json` - 训练集统计量
- `dataloader/dataloader.py` - 你的dataloader实现
- `dataloader/so101_tactile_dataset.py` - 你的dataset实现

### 文档
- `TREX_DATALOADER_INTEGRATION.md` - 本文档
- `QUICKSTART_TREX.md` - 快速开始
- `README_TREX.md` - 详细说明
- `TREX_2FINGER_SUMMARY.md` - 项目总结

### 自动化
- `run_trex_vqvae.sh` - 一键训练脚本

---

## ✅ 集成验证清单

- [x] T-Rex官方代码提取完成
- [x] 双指系统适配完成
- [x] 你的dataloader集成完成
- [x] 统计量计算脚本就绪
- [x] 训练脚本就绪
- [x] 推理脚本就绪
- [x] 配置文件就绪
- [x] 自动化脚本就绪
- [x] 文档完善

---

## 🎯 下一步

1. **运行快速测试** (推荐首先执行)
   ```bash
   python TactileSelfencoder/train_trex_vqvae.py \
       --config TactileSelfencoder/trex_2finger_test_config.yaml \
       --data_config dataloader/vqvae_tactile.yaml \
       --stats TactileSelfencoder/trex_tactile_stats.json \
       --output_dir outputs/trex_test
   ```

2. **检查测试结果**
   ```bash
   ls -lh outputs/trex_test/
   cat outputs/trex_test/training.log | tail -20
   ```

3. **运行完整训练**
   ```bash
   ./run_trex_vqvae.sh
   ```

4. **监控训练**
   ```bash
   tail -f outputs/trex_vqvae/training.log
   ```

5. **生成可视化**
   ```bash
   python TactileSelfencoder/inference_trex.py \
       --checkpoint outputs/trex_vqvae/best.pt \
       --data_config dataloader/vqvae_tactile.yaml \
       --output outputs/trex_vqvae/visualizations
   ```

---

## 💡 提示

- GPU推荐: NVIDIA GPU (>=6GB VRAM)
- 训练时间: 约2-4小时 (30 epochs, RTX 3090)
- 磁盘空间: 约500MB (checkpoints + visualizations)
- 内存需求: 约8GB RAM

---

**准备就绪！现在可以开始训练 T-Rex VQ-VAE 了！** 🚀
