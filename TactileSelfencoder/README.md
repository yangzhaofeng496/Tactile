# TactileSelfencoder 目录说明

本目录包含触觉编码器的实现，包括T-Rex官方VQ-VAE适配和其他实验性实现。

---

## 📁 目录结构

```
TactileSelfencoder/
├── trex_official/          # T-Rex官方VQ-VAE实现（未修改）
│   ├── encoder.py          - F6PerFingerEncoder
│   ├── decoder.py          - F6PerFingerDecoder  
│   ├── quantizer.py        - VQEMAQuantizer
│   ├── tactile_vqvae.py    - TactileVQVAE主模型
│   └── __init__.py         - 模块导出
│
├── trex_docs/              # 完整文档和测试
│   ├── README.md           - 文档索引（从这里开始）
│   ├── QUICKSTART_TREX.md  - 快速开始指南
│   ├── README_TREX.md      - 详细使用文档
│   ├── TREX_2FINGER_SUMMARY.md - 项目报告
│   ├── T-REX_OFFICIAL_ANALYSIS.md - 官方代码分析
│   ├── TREX_PROJECT_FILES.md - 文件清单
│   └── test_trex_model.py  - 单元测试
│
├── vqvae_checkpoints/      # 训练好的模型checkpoint
│   ├── so101/
│   ├── umi/
│   └── 50_16tokens_resume/
│
├── [T-Rex VQ-VAE 训练脚本]
│   ├── compute_trex_stats.py      - 计算数据统计量
│   ├── train_trex_vqvae.py        - 训练脚本
│   ├── inference_trex.py          - 推理和可视化
│   └── trex_2finger_config.yaml   - 训练配置
│
└── [其他实验性实现]
    ├── vqvae_model.py              - 原始VQ-VAE实现
    ├── train_vqvae.py              - 原始训练脚本
    ├── inference.py                - 原始推理
    └── vqvae_config.yaml           - 原始配置
```

---

## 🚀 快速开始（T-Rex VQ-VAE）

### 推荐路径：从文档开始

```bash
# 1. 阅读快速开始指南
cat TactileSelfencoder/trex_docs/QUICKSTART_TREX.md

# 2. 运行单元测试
python TactileSelfencoder/trex_docs/test_trex_model.py

# 3. 计算数据统计量
python TactileSelfencoder/compute_trex_stats.py \
    --data_config dataloader/vqvae_tactile.yaml \
    --output TactileSelfencoder/trex_tactile_stats.json

# 4. 开始训练
python TactileSelfencoder/train_trex_vqvae.py \
    --config TactileSelfencoder/trex_2finger_config.yaml \
    --data_config dataloader/vqvae_tactile.yaml \
    --stats TactileSelfencoder/trex_tactile_stats.json \
    --output_dir outputs/trex_vqvae
```

**或使用一键脚本**:
```bash
./run_trex_pipeline.sh
```

---

## 📚 文档入口

**主文档目录**: [trex_docs/README.md](trex_docs/README.md)

从那里可以找到：
- ✅ 快速开始指南
- ✅ 详细使用文档  
- ✅ 完整项目报告
- ✅ 官方代码分析
- ✅ 文件清单索引

---

## 🎯 主要实现对比

| 特性 | T-Rex Official | 原始实现 |
|------|----------------|----------|
| **Per-finger量化** | ✅ 是 | ❌ 否 |
| **Finger ID Embedding** | ✅ 是 | ❌ 否 |
| **Magnitude-weighted Loss** | ✅ 是 | ❌ 否 |
| **死亡码字复活** | ✅ 随机采样 | ⚠️ 简单实现 |
| **官方验证** | ✅ 已验证 | ❌ 实验性 |

**推荐使用**: T-Rex Official 实现

---

## 🔧 配置文件说明

| 文件 | 用途 | 推荐 |
|------|------|------|
| `trex_2finger_config.yaml` | T-Rex双指训练配置 | ✅ 推荐 |
| `trex_vqvae_config.yaml` | 实验性配置 | ⚠️ |
| `vqvae_config.yaml` | 原始配置 | ⚠️ |

---

## 📊 模型性能

### T-Rex VQ-VAE (推荐)

| 指标 | 值 |
|------|---|
| 参数量 | 896K (0.90M) |
| Codebook大小 | 64 |
| 输入 | [B, 16, 2, 6] |
| 输出 | [B, 2] (per-finger indices) |
| 重建MSE | < 0.02 (预期) |
| 码本利用率 | > 90% (预期) |

---

## ⚠️ 重要提示

1. **优先使用T-Rex官方实现** - 已经过验证和测试
2. **完整文档** - 所有文档在 `trex_docs/` 目录
3. **测试先行** - 训练前先运行 `test_trex_model.py`
4. **统计量必需** - 训练前必须先计算数据统计量

---

## 🔗 相关链接

- **T-Rex官方仓库**: https://github.com/ZhuoyangLiu2005/T-Rex
- **项目文档**: [trex_docs/README.md](trex_docs/README.md)
- **快速开始**: [trex_docs/QUICKSTART_TREX.md](trex_docs/QUICKSTART_TREX.md)

---

## 📝 版本历史

- **v1.0** (2026-08-12) - T-Rex VQ-VAE双指适配完成
  - 提取官方源码
  - 适配双指系统
  - 完整测试和文档

---

**维护状态**: ✅ 活跃维护  
**推荐使用**: T-Rex Official 实现  
**文档完整度**: 100%
