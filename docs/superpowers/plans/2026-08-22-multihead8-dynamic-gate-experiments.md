# Multi-Head 8 Dynamic Gate 实验计划

> 目标：以当前表现最好的 `multihead=8 + dynamic gate` 为基线，验证 head 专业化、模态门控稳定性和动作阶段依赖，而不是继续盲目增加模型容量。

## 实验原则

- 每轮只改变一个结构或超参数变量。
- episode 划分、缓存、随机种子、batch size 和训练轮数保持一致。
- 每个实验使用独立的 `checkpoint_dir`，不覆盖已有结果。
- checkpoint 只依据 validation loss 选择；test loss 只能用于最终诊断。
- 当前 `dataloader/tactile_dataloader.yaml` 是 `train=0.90、val=0.10、test=0.0`。正式比较前应固定一个有独立 test episode 的 split，例如 `0.80/0.10/0.10`；否则只能比较 train/val，不能宣称泛化提升。
- 当前训练窗口为每个 episode 随机采样 25%，所有实验必须保持 `train_window_fraction=0.25` 不变。

## B0：当前最佳基线

当前基线配置：

```yaml
fusion:
  type: multihead
  num_heads: 8
  use_gate: true
  use_modality_gate: true
  modality_gate_temperature: 1.5
  modality_gate_residual_scale: 0.5
```

记录：

- best validation loss 及对应 epoch；
- test loss（若启用独立 test split）；
- 8 个 head 的平均 gate 权重和使用率；
- 4 个模态的平均权重、标准差和熵；
- 每个 episode 的 loss 分布；
- 参数量、每 epoch 时间和显存。

## 实验目录和 W&B 命名

为避免覆盖，两个重点实验使用以下固定名称：

| 实验 | checkpoint 目录 | W&B run name |
|---|---|---|
| E3：8-head balance + diversity | `residual_actmodel/experiments/400_50hil_820_multihead8_gate_balance_diversity` | `multihead8_gate_balance_diversity_seed11` |
| E4：时间步级 modality gate | `residual_actmodel/experiments/400_50hil_820_timestep_modality_gate` | `multihead8_timestep_modality_gate_seed11` |

启动时也可以用命令行覆盖名称：

```bash
python train.py \
  --checkpoint-dir residual_actmodel/experiments/400_50hil_820_multihead8_gate_balance_diversity \
  --wandb-name multihead8_gate_balance_diversity_seed11
```

时间步级 gate 在 E3 完成并确认收益后再启动，避免同时改变路由损失和 gate 的时间结构。

## E1：Head 数量确认

目的：确认 8 个 head 的优势来自真实分工，还是单纯来自更多参数。

| 实验 | `num_heads` | 其他设置 |
|---|---:|---|
| E1-4 | 4 | 完全同 baseline |
| E1-8 | 8 | baseline |
| E1-12 | 12 | `hidden_dim=96`，其余不变 |

注意：保持 `hidden_dim=96` 时，head 数量变化也会改变参数量；因此同时记录参数量。若 E1-12 只略好但参数增加明显，不优先采用。

判据：验证集 loss、独立 test loss、每个 head 的使用率和 head 输出相似度共同判断。

## E2：Modality Gate 稳定性

目的：验证残差式门控和 temperature 是否真的优于直接缩放。

固定 `num_heads=8`、`use_gate=true`、`use_modality_gate=true`，只改变：

| 实验 | temperature | residual scale |
|---|---:|---:|
| E2-a | 1.0 | 0.5 |
| E2-b | 1.5 | 0.5（当前） |
| E2-c | 2.0 | 0.5 |
| E2-d | 1.5 | 0.25 |
| E2-e | 1.5 | 1.0 |

记录 modality gate 的熵和最小权重。若 gate 很快变成近似 one-hot，同时 val/test 变差，说明门控过早塌缩。

## E3：Head 负载均衡与专业化

目的：让 8 个 head 真正分工，避免少数 head 垄断路由。

固定所有 baseline 设置，只增加辅助损失：

```text
L = L_action + λ_balance L_balance + λ_diversity L_diversity
```

建议实验：

| 实验 | `lambda_balance` | `lambda_diversity` |
|---|---:|---:|
| E3-a | 0 | 0 |
| E3-b | 0.001 | 0 |
| E3-c | 0.005 | 0 |
| E3-d | 0.001 | 0.001 |

其中：

```python
mean_gate = gate_weights.mean(dim=0)
L_balance = num_heads * (mean_gate.square().sum())
```

`L_balance` 只约束 batch 平均使用率，不要求每个样本都平均使用 head。`L_diversity` 可使用不同 head 输出之间的 cosine similarity 惩罚。

参考：稀疏 MoE 工作把路由负载均衡作为防止专家塌缩的重要机制，[Sparsely-Gated MoE](https://arxiv.org/abs/1701.06538) 和 [Switch Transformer](https://arxiv.org/abs/2101.03961)。

## E4：窗口级 gate 与动作阶段 gate

目的：验证“不同动作阶段依赖不同模态”的假设。

当前 gate 输出：

```text
[B, 4]
```

候选升级版本：

```text
[B, 30, 4]
```

让每个 action horizon step 产生一组 modality 权重。建议先只做 modality gate 的时间展开，暂时保留 8 个 head 的窗口级 gate，降低变量数量：

```text
encoded modalities
    ↓
decoder/action queries
    ↓
per-step modality gate [B, 30, 4]
    ↓
action residual [B, 30, 6]
```

比较：

- E4-a：当前窗口级 gate；
- E4-b：每 30 个动作步一组 gate。

同时绘制每个动作步的四模态权重热图。该设计和 Perceiver IO 的 query-specific structured output 思路一致，可参考 [Perceiver IO](https://arxiv.org/abs/2107.14795)。

## E5：条件调制而不是只做模态缩放

目的：验证当前力和状态是否应该调制融合特征，而不只是参与 gate。

固定最佳 E1–E4 配置，比较：

| 实验 | 条件调制 |
|---|---|
| E5-a | 关闭 `force_film` |
| E5-b | 开启当前 `force_film` |
| E5-c | 用 `state + current_force` 共同生成 FiLM 参数 |

FiLM 的核心是由条件输入生成 feature-wise 的 `gamma` 和 `beta`，可参考 [FiLM](https://arxiv.org/abs/1709.07871)。

因为历史实验显示 FiLM 可能受数据覆盖限制，本实验必须放在 gate 稳定性实验之后，并且只接受验证集和独立 test 同时改善的结果。

## 结果判定

每个实验至少记录：

- best val loss；
- validation-selected checkpoint 的 test loss；
- train/val/test 曲线；
- episode-level loss；
- head gate 使用率和熵；
- modality gate 使用率和熵；
- 参数量、训练时间、显存。

优先级规则：

1. 先看 validation-selected checkpoint 的独立 test loss；
2. test 改善幅度小于多 seed 波动时，不认为结构有效；
3. 若 train/val 变好但 test 变差，判定为过拟合而不是提升；
4. 最佳配置使用第二个 split seed 复现；
5. 连续三轮单变量实验没有稳定收益后，停止继续增加结构复杂度，转向数据覆盖和 episode 分布分析。

## 推荐执行顺序

```text
B0 baseline
  ↓
E1 head 数量确认
  ↓
E2 temperature / residual scale
  ↓
E3 balance + diversity loss
  ↓
E4 [B,4] → [B,30,4] 时间步级 gate
  ↓
E5 state/force 条件 FiLM
  ↓
第二个 seed 复现最佳配置
```
