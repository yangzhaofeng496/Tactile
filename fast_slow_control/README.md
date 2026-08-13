# 快慢模型协调控制器（硬件版本）

硬件接入版本的快慢模型协调控制器，支持真实机器人硬件部署。

## 架构概述

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   摄像头     │     │  触觉传感器  │     │ 机器人控制器 │
│   30Hz      │     │   90Hz      │     │             │
└──────┬──────┘     └──────┬──────┘     └──────▲──────┘
       │                   │                   │
       └───────┬───────────┘                   │
               │                               │
       ┌───────▼────────┐                      │
       │ HardwareSensor │                      │
       │    Manager     │                      │
       └───────┬────────┘                      │
               │                               │
       ┌───────▼────────┐                      │
       │  ACT 慢模型    │                      │
       │  (30Hz 推理)   │                      │
       └───────┬────────┘                      │
               │                               │
               │  基准动作                      │
               │                               │
       ┌───────▼────────┐                      │
       │  残差快模型     │                      │
       │  (90Hz 修正)   │                      │
       └───────┬────────┘                      │
               │                               │
               └───────────────────────────────┘
                     最终动作
```

## 硬件接口

本系统预留了三个硬件接口文件，位于 `hardwareInterface/` 目录下，需要根据实际硬件实现：

### 1. `hardwareInterface/camera_interface.py` - 摄像头接口

**需要实现的类**: `CameraInterface`

**主要方法**:
- `__init__(camera_hz, **kwargs)`: 初始化摄像头
- `start()`: 启动摄像头采集
- `stop()`: 停止摄像头采集
- `get_latest_frame()`: 获取最新帧（返回 np.ndarray 或 None）
- `is_ready()`: 检查摄像头是否就绪

**实现要点**:
- 返回的图像可以是原始图像 `[H, W, C]` 或预处理后的特征向量
- 需要处理线程安全（如果使用异步采集）
- 建议实现帧缓冲避免数据丢失

### 2. `hardwareInterface/tactile_interface.py` - 触觉传感器接口

**需要实现的类**: `TactileInterface`

**主要方法**:
- `__init__(tactile_hz, tactile_dim, **kwargs)`: 初始化触觉传感器
- `start()`: 启动触觉采集
- `stop()`: 停止触觉采集
- `get_latest_reading()`: 获取最新触觉数据（返回 np.ndarray 或 None）
- `is_ready()`: 检查传感器是否就绪

**实现要点**:
- 返回形状为 `[tactile_dim]` 的向量（默认12维：左右各6维）
- 高频采集（90Hz），需要注意实时性
- 建议使用高优先级线程或实时调度

### 3. `hardwareInterface/robot_interface.py` - 机器人控制器接口

**需要实现的类**: `RobotInterface`

**主要方法**:
- `__init__(action_dim, **kwargs)`: 初始化机器人控制器
- `connect()`: 连接到机器人
- `disconnect()`: 断开连接
- `enable()`: 使能机器人
- `disable()`: 失能机器人
- `send_action(action)`: 发送动作指令（返回 bool）
- `get_state()`: 获取机器人当前状态
- `is_connected()`: 检查连接状态

**实现要点**:
- `action` 形状为 `[action_dim]`（默认6维关节角度/速度）
- 需要实现安全检查（急停、限位等）
- 建议实现心跳检测和超时保护

## 使用方法

### 1. 实现硬件接口

在三个接口文件中实现真实硬件的连接逻辑：

```python
# hardwareInterface/camera_interface.py
class RealCameraInterface(CameraInterface):
    def __init__(self, camera_hz=30, device_id=0):
        import cv2
        self.camera_hz = camera_hz
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FPS, camera_hz)
        # ... 更多初始化
    
    # 实现其他抽象方法
```

### 2. 修改控制器初始化

在 `fast_slow_control.py` 的 `FastSlowController.__init__()` 中，将 Mock 实现替换为真实实现：

```python
# 替换这部分代码：
# camera = MockCameraInterface(...)
# tactile = MockTactileInterface(...)
# robot = MockRobotInterface(...)

# 改为：
from hardwareInterface.camera_interface import RealCameraInterface
from hardwareInterface.tactile_interface import RealTactileInterface
from hardwareInterface.robot_interface import RealRobotInterface

camera = RealCameraInterface(camera_hz=self.camera_hz)
tactile = RealTactileInterface(tactile_hz=self.tactile_hz)
robot = RealRobotInterface(action_dim=self.state_dim)
```

### 3. 配置参数

编辑 `config.yaml` 调整硬件和模型参数：

```yaml
sensors:
  camera_hz: 30          # 摄像头频率
  tactile_hz: 90         # 触觉传感器频率
  tactile_dim: 12        # 触觉向量维度
  state_dim: 6           # 关节状态维度

act:
  horizon: 30            # ACT 每次推理步数
  control_hz: 10         # 基准控制频率
  pretrained_path: "/path/to/act/model"
  device: "cuda"

residual:
  control_hz: 90         # 残差网络控制频率
  tactile_history: 15    # 触觉历史长度
  checkpoint_path: "/path/to/residual/model"
  device: "cuda"
```

### 4. 运行控制器

**模式1（不带残差）**：
```bash
python fast_slow_control/fast_slow_control.py --mode 1 --config fast_slow_control/config.yaml
```

**模式2（带残差修正）**：
```bash
python fast_slow_control/fast_slow_control.py --mode 2 --config fast_slow_control/config.yaml
```

## 控制模式说明

### Mode 1: 不带残差

- ACT 模型以 30Hz 推理，输出 30 步动作块
- 以 10Hz 频率直接发送 ACT 动作到机器人
- 适合：环境变化慢、ACT 模型精度高的场景

### Mode 2: 带残差修正

- ACT 模型提供基准动作（30Hz 推理）
- 残差网络使用触觉反馈以 90Hz 频率修正动作
- 最终以 90Hz 发送修正后的动作到机器人
- 适合：接触任务、需要高频触觉反馈的场景

## 安全注意事项

1. **急停机制**: 确保 `robot_interface.py` 实现了急停功能
2. **限位保护**: 在发送动作前检查关节限位
3. **心跳检测**: 实现控制器超时保护
4. **渐进启动**: 首次运行建议降低速度和行程
5. **监控日志**: 关注控制频率是否达到预期

## 测试流程

### 1. 单元测试（使用 Mock 接口）

```bash
# 使用内置 Mock 实现测试控制逻辑
python fast_slow_control/fast_slow_control.py --mode 1 --config config.yaml
```

### 2. 硬件接口测试

分别测试三个硬件接口是否正常工作：

```python
# 测试摄像头
from hardwareInterface.camera_interface import RealCameraInterface
camera = RealCameraInterface(camera_hz=30)
camera.start()
frame = camera.get_latest_frame()
print(f"Frame shape: {frame.shape}")
camera.stop()
```

或运行完整的测试套件：

```bash
cd fast_slow_control
python test_hardware_interfaces.py
```

### 3. 集成测试

使用真实硬件运行完整控制循环，建议：
- 先以低速运行（`speedup: 0.1`）
- 逐步提高到实时（`speedup: 1.0`）
- 监控各传感器数据质量和控制频率

## 故障排查

### 问题：控制频率达不到预期

- 检查传感器采样频率是否配置正确
- 检查模型推理时间（GPU 是否正常工作）
- 降低 `residual.control_hz` 以匹配实际硬件能力

### 问题：动作抖动

- 检查触觉数据质量（噪声过大？）
- 调整残差网络的修正幅度
- 增加 `tactile_history` 长度做平滑

### 问题：硬件连接失败

- 检查设备权限（串口、USB、网络）
- 检查 IP 地址、端口配置
- 查看具体硬件驱动文档

## 性能优化建议

1. **模型推理**: 使用 TensorRT 或 ONNX 加速
2. **数据传输**: 使用共享内存避免拷贝
3. **线程优先级**: 提高控制线程优先级
4. **预分配内存**: 避免运行时动态分配

## 文件结构

```
fast_slow_control/
├── fast_slow_control.py      # 主控制器
├── camera_interface.py        # 摄像头接口（待实现）
├── tactile_interface.py       # 触觉接口（待实现）
├── robot_interface.py         # 机器人接口（待实现）
├── config.yaml               # 配置文件
└── README.md                 # 本文档
```

## 联系与支持

如有问题请联系项目维护者或查看项目文档。
