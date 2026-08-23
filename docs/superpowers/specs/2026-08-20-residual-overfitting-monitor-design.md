# Residual ACT 过拟合监测与自动实验设计

## 目标

为 residual ACT 训练增加可审计的 test-loss 监测、epoch 边界自动停训和单变量根因实验流程，避免把验证集或测试集错误地用于参数更新或训练决策。

## 当前基线

- 训练入口：`train.py`
- 数据入口：`dataloader/cache_loader.py`
- 当前缓存：`outputs/act_cache/act_cache_450.pt`
- 当前模型可训练参数：约 253,204
- 当前训练数据按 episode 划分为 train/val/test，比例为 70/15/15，seed 为 42
- 已观察到 train/val 继续下降而 test loss 在较早 epoch 后回升

## 设计

### 1. 评估隔离

- train 阶段允许更新参数。
- val 阶段只执行 `eval()`、`no_grad()`，不得调用 optimizer/scaler 更新参数。
- test 阶段只执行无梯度评估。
- test loss 不参与 dropout、学习率、模型结构或 checkpoint 选择。
- checkpoint 选择只依据 val loss；最终 test 只作为泛化报告。

### 2. 过拟合监测

每个 epoch 记录：`train_loss`、`val_loss`、`test_loss`、当前最佳 test loss、当前最佳 val loss。

默认触发条件：

```text
当前 test loss 连续 3 个 epoch 上升
且当前 test loss > 历史最低 test loss * 1.01
且 train 或 val 在相邻窗口仍有下降
```

触发时只在当前 epoch 完成后停止，保存：

- 当前模型 checkpoint；
- 最佳 val checkpoint；
- loss CSV；
- loss 曲线 PNG；
- 触发原因和配置快照。

### 3. 根因实验顺序

每轮只改变一个变量，并使用独立 checkpoint/output 目录：

1. 评估隔离修复：确认 val/test 不更新参数。
2. 数据完整性：检查 cache key、episode 边界、train/val/test episode 是否重叠。
3. 窗口冗余：比较原始滑动窗口与增大 stride/每 episode 限制窗口。
4. 输入依赖：比较 action-only、state/current-force-only 和完整模型。
5. 目标与 baseline：比较 ACT action baseline 与 residual target 的难度。
6. 正则化：比较参数规模、weight decay、action noise 和 early stopping。
7. 分布差异：按 episode 统计 test loss，定位是否由少数 episode 或任务状态造成。

每轮实验保存配置、参数量、cache 路径、split seed、训练曲线和停止原因。

### 4. 结论规则

- 若修复评估隔离后 test 不再持续恶化，则根因为评估流程污染。
- 若减少窗口冗余后泛化明显改善，则根因主要是时间窗口相关性过高。
- 若 action-only 接近完整模型，则 residual 主要依赖 ACT action，触觉/状态分支贡献有限。
- 若所有单变量实验仍出现稳定的 train/val 下降、test 上升，则结论指向 episode 分布差异、目标不可泛化或数据覆盖不足，而不是单纯模型容量过大。
- 连续三轮有证据的修改未改善时，停止继续盲目调参，形成架构层结论。

### 5. Gmail 报告

实验结束后发送一封总结邮件到当前认证 Gmail 账户，包含：

- 实验列表和配置差异；
- 每轮最佳 train/val/test 指标；
- 过拟合触发 epoch；
- 根因证据；
- 最终建议；
- 本地曲线和 CSV 文件路径。

## 安全边界

- 不删除原始 cache、checkpoint 或日志。
- 不覆盖已有实验目录；每轮使用新目录。
- 不在训练未到 epoch 边界时强制终止，除非进程发生异常。
- 不把 test loss 用于调参或选择最终 checkpoint。
