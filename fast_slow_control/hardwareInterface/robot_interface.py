#!/usr/bin/env python3
"""
机器人控制器接口 - 待实现

请实现 RobotInterface 类来接入真实机器人控制器硬件。
"""

import numpy as np
from abc import ABC, abstractmethod


class RobotInterface(ABC):
    """机器人控制器硬件接口抽象类。

    请继承此类并实现所有抽象方法来对接真实机器人控制器硬件。
    """

    @abstractmethod
    def __init__(self, action_dim: int = 6, **kwargs):
        """初始化机器人控制器。

        Args:
            action_dim: 动作维度（例如：6自由度机械臂 = 6维）
            **kwargs: 其他硬件特定参数（如IP地址、端口、控制模式等）
        """
        pass

    @abstractmethod
    def connect(self) -> None:
        """连接到机器人控制器。"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """断开机器人控制器连接。"""
        pass

    @abstractmethod
    def send_action(self, action: np.ndarray) -> bool:
        """发送控制指令到机器人。

        Args:
            action: 动作向量，形状为 [action_dim]

        Returns:
            bool: True 表示发送成功
        """
        pass

    @abstractmethod
    def get_state(self) -> np.ndarray | None:
        """获取机器人当前状态。

        Returns:
            np.ndarray: 状态向量（如关节位置、速度等），形状为 [state_dim]
            None: 如果无法获取状态
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """检查是否已连接到机器人。

        Returns:
            bool: True 表示连接正常
        """
        pass

    @abstractmethod
    def enable(self) -> None:
        """使能机器人（进入可控制状态）。"""
        pass

    @abstractmethod
    def disable(self) -> None:
        """失能机器人（安全停止）。"""
        pass


# ============================================================================
# 实现示例（请根据实际硬件替换）
# ============================================================================

class MockRobotInterface(RobotInterface):
    """模拟机器人控制器实现 - 仅供测试使用。

    实际部署时请替换为真实硬件实现。
    """

    def __init__(self, action_dim: int = 6, **kwargs):
        self.action_dim = action_dim
        self._connected = False
        self._enabled = False
        self._state = np.zeros(action_dim, dtype=np.float32)

    def connect(self) -> None:
        self._connected = True
        print("[MockRobot] 已连接到机器人控制器")

    def disconnect(self) -> None:
        self._connected = False
        self._enabled = False
        print("[MockRobot] 已断开机器人控制器连接")

    def send_action(self, action: np.ndarray) -> bool:
        if not self._connected or not self._enabled:
            return False
        # 模拟发送动作
        rounded = ", ".join(f"{v:.3f}" for v in action)
        print(f"[MockRobot] 发送动作: [{rounded}]")
        # 更新内部状态模拟
        self._state = 0.9 * self._state + 0.1 * action
        return True

    def get_state(self) -> np.ndarray | None:
        if not self._connected:
            return None
        return self._state.copy()

    def is_connected(self) -> bool:
        return self._connected

    def enable(self) -> None:
        if self._connected:
            self._enabled = True
            print("[MockRobot] 机器人已使能")

    def disable(self) -> None:
        self._enabled = False
        print("[MockRobot] 机器人已失能")
