#!/usr/bin/env python3
"""
触觉传感器接口 - 待实现

请实现 TactileInterface 类来接入真实触觉传感器硬件。
"""

import numpy as np
from abc import ABC, abstractmethod


class TactileInterface(ABC):
    """触觉传感器硬件接口抽象类。

    请继承此类并实现所有抽象方法来对接真实触觉传感器硬件。
    """

    @abstractmethod
    def __init__(self, tactile_hz: int = 90, tactile_dim: int = 12, **kwargs):
        """初始化触觉传感器。

        Args:
            tactile_hz: 触觉采样频率（Hz）
            tactile_dim: 触觉数据维度（例如：左右各6维 = 12维）
            **kwargs: 其他硬件特定参数（如串口、设备地址等）
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """启动触觉传感器采集。"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """停止触觉传感器采集。"""
        pass

    @abstractmethod
    def get_latest_reading(self) -> np.ndarray | None:
        """获取最新的触觉读数。

        Returns:
            np.ndarray: 触觉数据，形状为 [tactile_dim]
            None: 如果当前没有新数据
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """检查触觉传感器是否就绪。

        Returns:
            bool: True 表示传感器正常工作
        """
        pass


# ============================================================================
# 实现示例（请根据实际硬件替换）
# ============================================================================

class MockTactileInterface(TactileInterface):
    """模拟触觉传感器实现 - 仅供测试使用。

    实际部署时请替换为真实硬件实现。
    """

    def __init__(self, tactile_hz: int = 90, tactile_dim: int = 12, **kwargs):
        self.tactile_hz = tactile_hz
        self.tactile_dim = tactile_dim
        self._running = False
        self._rng = np.random.default_rng(1)
        self._t = 0.0

    def start(self) -> None:
        self._running = True
        self._t = 0.0
        print(f"[MockTactile] 启动触觉传感器 @ {self.tactile_hz}Hz")

    def stop(self) -> None:
        self._running = False
        print("[MockTactile] 停止触觉传感器")

    def get_latest_reading(self) -> np.ndarray | None:
        if not self._running:
            return None
        # 模拟触觉信号：正弦波 + 噪声
        self._t += 1.0 / self.tactile_hz
        raw = np.stack(
            [
                np.sin(2 * np.pi * 0.5 * self._t + i)
                + 0.1 * self._rng.standard_normal(self.tactile_dim)
                for i in range(6)
            ],
            axis=0,
        ).mean(axis=0)
        return raw.astype(np.float32)

    def is_ready(self) -> bool:
        return self._running
