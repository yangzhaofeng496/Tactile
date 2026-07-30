# deform_autodecoder

基于 `architecture/architecture_deformation.drawio` 重建的形变图自编码器。

## 输入输出

- 输入: `[B, C, 240, 240]`
- latent: `[B, 128, 15, 15]`
- 重建: `[B, C, 240, 240]`

## 文件

- `model.py`: 模型与重建损失
- `dataloader.py`: 自编码器专用 LeRobot dataloader
- `config.yaml`: 模型和训练配置
- `load_model.py`: 按配置创建模型
- `train.py`: 最小训练入口

## 训练数据格式

`train.py` 支持两种数据来源：

- `source: tensor`
- `source: dataloader`

当 `source: tensor` 时，读取 `torch.load(path)` 得到的张量，支持两种格式：

- `[N, 240, 240]`
- `[N, 1, 240, 240]`

配置文件中设置：

```yaml
data:
  train_tensor_path: /abs/path/train_deform.pt
  val_tensor_path: /abs/path/val_deform.pt
```

当 `source: dataloader` 时，复用项目根目录的 [dataloader/dataloader.py](/home/yang/TactileEncoder/dataloader/dataloader.py) 和对应 YAML。`train.py` 会从 batch 的 `tactile_history` 中：
当 `source: dataloader` 时，使用本目录下的 [dataloader.py](/home/yang/TactileEncoder/deform_autodecoder/dataloader.py)。它会：

- 从 LeRobot 数据集读取 `tactile_image` 历史帧
- 按通道拼接多个触觉图像键
- 用 `history_index` 选当前或历史帧
- 用 `channel_start` 和 `num_channels` 抽出连续通道
- 按需 resize 到 `target_size`

训练阶段直接消费 batch 里的 `image` 字段，形状为 `[B, C, H, W]`。

## 运行

```bash
python3 deform_autodecoder/load_model.py
python3 deform_autodecoder/train.py
```
