# T-Rex VQ-VAE 快速参考

## 🚀 一键启动
```bash
./run_trex_vqvae.sh
```

## 📝 分步执行

### 1️⃣ 计算统计量
```bash
python TactileSelfencoder/compute_trex_stats.py \
    --data_config dataloader/vqvae_tactile.yaml \
    --output TactileSelfencoder/trex_tactile_stats.json
```

### 2️⃣ 训练模型
```bash
# 完整训练 (30 epochs)
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/trex_vqvae

# 快速测试 (3 epochs)
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_test_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/trex_test
```

### 3️⃣ 推理可视化
```bash
python TactileSelfencoder/inference_trex.py \
    --checkpoint outputs/trex_vqvae/best.pt \
    --data_config dataloader/vqvae_tactile.yaml \
    --output outputs/trex_vqvae/visualizations
```

## 📊 监控训练
```bash
# 实时查看日志
tail -f outputs/trex_vqvae/training.log

# 查看最近的训练指标
tail -20 outputs/trex_vqvae/training.log

# 检查checkpoint
ls -lh outputs/trex_vqvae/*.pt
```

## 🔍 关键文件

| 文件 | 说明 |
|------|------|
| `trex_2finger_config.yaml` | 完整训练配置 (30 epochs) |
| `trex_2finger_test_config.yaml` | 快速测试配置 (3 epochs) |
| `dataloader/vqvae_tactile.yaml` | 你的数据配置 |
| `trex_tactile_stats.json` | 训练集统计量 |
| `outputs/trex_vqvae/best.pt` | 最佳模型 |
| `outputs/trex_vqvae/latest.pt` | 最新模型 |

## 📈 预期指标

| 指标 | 目标值 |
|------|--------|
| 重建损失 | < 0.02 |
| 码本利用率 | > 90% |
| 困惑度 | 50-60 |

## 🎯 数据流

```
Dataloader输出
    ↓ [B, 16, 12]
Reshape
    ↓ [B, 16, 2, 6]
标准化
    ↓ normalized
T-Rex Encoder
    ↓ [B, 2, 256]
VQ Quantizer
    ↓ [B, 2] indices
T-Rex Decoder
    ↓ [B, 16, 2, 6]
重建输出
```

## 💡 常用技巧

### 恢复训练
```bash
# 自动从 latest.pt 恢复
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/trex_vqvae  # 相同目录
```

### 修改batch size
编辑 `trex_2finger_config.yaml`:
```yaml
train:
  batch_size: 256  # 默认128
```

### 修改码本大小
编辑 `trex_2finger_config.yaml`:
```yaml
model:
  codebook_size: 128  # 默认64
```

### 调整学习率
编辑 `trex_2finger_config.yaml`:
```yaml
train:
  lr: 5.0e-4  # 默认3.0e-4
```

## 📚 详细文档

- **集成说明**: `TREX_DATALOADER_INTEGRATION.md`
- **快速开始**: `QUICKSTART_TREX.md`
- **详细使用**: `README_TREX.md`
- **项目总结**: `TREX_2FINGER_SUMMARY.md`
- **官方分析**: `T-REX_OFFICIAL_ANALYSIS.md`

## 🐛 故障排查

### GPU内存不足
```yaml
# 减小batch size
train:
  batch_size: 64  # 或32
```

### 训练不稳定
```yaml
# 降低学习率
train:
  lr: 1.0e-4

# 增加warmup
train:
  warmup_steps: 1000
```

### 码本利用率低
```yaml
# 调整dead code参数
model:
  revive_freq: 50       # 更频繁检查
  revive_threshold: 0.5 # 更宽松阈值
```

---

**准备就绪！🎉**
