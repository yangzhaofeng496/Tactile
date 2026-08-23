# Visual Token 提取流程提示词

## 任务

从冻结的 ACT（Action Chunking Transformer）策略中提取视觉 token，供残差动作网络使用。

## 背景配置

- ACT：LeRobot 实现，`modeling_act.py`，vision_backbone=resnet18，dim_model=512
- 3 个相机：`wrist_cam`、`top`、`side`，输入 480×640
- 训练时推理用 `policy.predict_action_chunk(preprocessed_obs)`，观测经 preprocessor 归一化
- `robot_state_feature = STATE(6)`，`env_state_feature = None`
- 本项目 `use_act_visual=True`，输出存 `act_visual: [512]` 到 cache

## 提取步骤

1. **注册 hook 捕获 encoder 输出**：
   - 获取 ACT 模型：`policy.model.encoder`
   - `encoder.register_forward_hook(hook)`，hook 记录模块输出 `encoder_out`
   - 每次前向后立即读取并重置为 None（batch 循环中）

2. **得到 encoder 输出**：
   - shape = `[S, B, D]`，D = dim_model = 512
   - token 序列结构 = `[latent(1), robot_state(1), (env_state), 视觉token...]`

3. **计算前缀 token 数**：
   - `n_pre_tokens = 1`（latent）
   - 若 `robot_state_feature` 存在：`+1`
   - 若 `env_state_feature` 存在：`+1`
   - 本项目 = 2

4. **截取视觉 token**：
   - `visual_tokens = encoder_out[n_pre_tokens:, :, :].transpose(0, 1)` → `[B, S_vis, D]`

5. **mean pooling 聚合**：
   - `visual = visual_tokens.mean(dim=1)` → `[B, D]`
   - 对 token 维度做算术平均，每个样本的 S_vis 个 token 平均成一个 512 维向量
   - **实测 shape**（本项目配置）：encoder_out `[902, B, 512]`，n_pre=2，视觉 token `[900, B, 512]`，每相机 300 个空间 token，3 相机共 900 个

6. **存入缓存**：每窗口存 `act_visual: [512]`，配合 `act_chunk: [30, 6]` 等

## 残差模型侧消费

（`model.py` `VisualTokenEncoder`）

- 输入 `[B, D]`（已 pooled）或 `[B, S, D]`（未 pooled）
- 流程：`Linear(512→256) → LayerNorm → ReLU`，输出 256 维进 fusion

## 关键实现细节

- backbone 用 `IntermediateLayerGetter` 取 resnet18 `layer4`，输出 `[B, 512, 15, 20]`（480×640 输入，下采样 32 倍，15×20=300 个空间位置）
- 每个空间像素视为一个 token：`einops.rearrange("b c h w -> (h w) b c")`
- 位置编码：1D token 用可学习 Embedding，视觉 token 用 `ACTSinusoidalPositionEmbedding2d(dim_model//2)`（2D 正弦）
- 所有 token 一起过自注意力，视觉 token 之间、与 latent/state 之间互相 attend
- `encoder_img_feat_input_proj` 为 1×1 Conv2d 把 512 通道映射到 dim_model
- 实际运行时需先 `export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"` 以加载 torchcodec
