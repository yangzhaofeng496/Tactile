#!/usr/bin/env python3
"""
摄像头接口 - 待实现

请实现 CameraInterface 类来接入真实摄像头硬件。
"""

import numpy as np
from abc import ABC, abstractmethod


class CameraInterface(ABC):
    """摄像头硬件接口抽象类。

    请继承此类并实现所有抽象方法来对接真实摄像头硬件。
    """

    @abstractmethod
    def __init__(self, camera_hz: int = 30, **kwargs):
        """初始化摄像头。

        Args:
            camera_hz: 摄像头采样频率（Hz）
            **kwargs: 其他硬件特定参数（如设备ID、分辨率等）
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """启动摄像头采集。"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止摄像头采集。"""
        pass

    @abstractmethod
    def get_latest_frame(self) -> np.ndarray | None:
        """获取最新的摄像头帧。

        Returns:
            np.ndarray: 图像数据，形状为 [H, W, C] 或处理后的特征向量
            None: 如果当前没有新帧
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """检查摄像头是否就绪。

        Returns:
            bool: True 表示摄像头正常工作
        """
        pass


# ============================================================================
# 实现示例（请根据实际硬件替换）
# ============================================================================

class MockCameraInterface(CameraInterface):
    """模拟摄像头实现 - 仅供测试使用。

    实际部署时请替换为真实硬件实现。
    """

    def __init__(self, camera_hz: int = 30, state_dim: int = 6, **kwargs):
        self.camera_hz = camera_hz
        self.state_dim = state_dim
        self._running = False
        self._rng = np.random.default_rng(0)
        self._state = np.zeros(state_dim, dtype=np.float32)

    def start(self) -> None:
        self._running = True
        print(f"[MockCamera] 启动摄像头 @ {self.camera_hz}Hz")

    def stop(self) -> None:
        self._running = False
        print("[MockCamera] 停止摄像头")

    def get_latest_frame(self) -> np.ndarray | None:
        if not self._running:
            return None
        # 模拟图像观测：用 state 向量代替真实图像
        self._state = 0.98 * self._state + 0.02 * self._rng.standard_normal(
            self.state_dim
        ).astype(np.float32)
        return self._state.copy()

    def is_ready(self) -> bool:
        return self._running
