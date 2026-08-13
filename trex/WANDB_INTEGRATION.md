# T-Rex VQ-VAE Wandb Integration

## 功能

T-Rex 训练脚本现在支持 Wandb 日志记录和可视化，与 `TactileSelfencoder/train_vqvae.py` 功能一致。

## 记录内容

### 训练指标（每 N 个 batch）
- `train/batch_loss` - 总损失
- `train/batch_recon_loss` - 重建损失
- `train/batch_vq_loss` - VQ 损失
- `train/batch_perplexity` - 码本复杂度
- `train/batch_active_codes` - 活跃码本数量
- `train/lr` - 学习率

### 每个 Epoch 指标
- `train/epoch_*` - 训练集平均指标
- `val/epoch_*` - 验证集平均指标
- `train/epoch_revived` - 本 epoch 复活的码本数量

### 簇可视化（每个 epoch 结束）
- **PCA 散点图**: 显示编码器输出和码本中心在 PCA 空间的分布
  - 不同的码本用不同颜色表示
  - 不同的手指用不同的标记形状表示（圆形、方形等）
  - 星形标记显示码本中心位置
- **使用频率条形图**: 显示每个码本的使用次数
- **统计信息**:
  - `active_codes`: 被使用的码本数量
  - `perplexity`: 码本使用的均匀程度（越高越好）
  - `max_usage_fraction`: 最常用码本的使用比例

## 配置

在 `trex_2finger_config.yaml` 中配置：

```yaml
wandb:
  enabled: true              # 启用/禁用 wandb
  project: trex-vqvae        # wandb 项目名
  run_name: null             # null = 自动生成名称
  tags:                      # 标签
    - trex
    - 2-finger
    - vqvae

train:
  wandb_log_every: 50        # 每 50 个 batch 记录一次
```

## 使用方法

### 启用 Wandb
```bash
cd /home/yang/TactileEncoder
./trex/run_trex_vqvae.sh
```

确保配置文件中 `wandb.enabled: true`。

### 禁用 Wandb
将配置文件中设置为：
```yaml
wandb:
  enabled: false
```

## 可视化特点

T-Rex 的簇可视化与 `TactileSelfencoder` 的主要区别：

1. **Per-finger 粒度**: T-Rex 使用 per-finger 量化，所以每个样本有 2 个码本索引
2. **多标记**: 使用不同形状的标记区分不同手指的数据点
3. **统计信息**: 统计量是跨所有手指计算的

## 输出示例

训练时会在 Wandb 看到：
- 实时训练曲线（loss、perplexity、active codes）
- 每个 epoch 的簇分布可视化
- 训练集和验证集的分别可视化
- 码本使用频率分布

这些可视化帮助诊断：
- 码本崩塌（只有少数码本被使用）
- 训练动态（码本如何被分配和使用）
- Per-finger 的量化模式
