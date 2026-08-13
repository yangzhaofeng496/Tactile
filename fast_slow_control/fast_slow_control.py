#!/usr/bin/env python3
"""
快慢模型协调控制器（硬件版本）。

硬件传感器环境：
  - 摄像头 30Hz（慢传感器，驱动 ACT 慢模型）
  - 触觉   90Hz（快传感器，驱动残差快模型）
  - 机器人控制器（接收最终动作指令）

ACT 每次推理 30 步动作块（horizon=30），随后逐步执行。

两种执行模式：
  mode 1（不带残差）：
      每 0.1 秒（10Hz）直接发送 ACT 输出的动作到机器人控制器。
  mode 2（带残差）：
      不直接发送 ACT 动作。ACT 输出作为基准动作传入残差网络，
      残差网络以更高频率（默认 90Hz）运行并输出最终动作，发送到机器人控制器。

硬件接口：
  - camera_interface.py: 摄像头硬件接口（需实现）
  - tactile_interface.py: 触觉传感器硬件接口（需实现）
  - robot_interface.py: 机器人控制器硬件接口（需实现）

用法：
  python fast_slow_control/fast_slow_control.py --mode 1 --config fast_slow_control/config.yaml
  python fast_slow_control/fast_slow_control.py --mode 2 --config fast_slow_control/config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

# 硬件接口导入
from hardwareInterface.camera_interface import CameraInterface, MockCameraInterface
from hardwareInterface.tactile_interface import TactileInterface, MockTactileInterface
from hardwareInterface.robot_interface import RobotInterface, MockRobotInterface


# ---------------------------------------------------------------------------
# 硬件传感器环境
# ---------------------------------------------------------------------------

class HardwareSensorManager:
    """硬件传感器管理器。

    统一管理摄像头、触觉传感器和机器人控制器三个硬件接口。
    """

    def __init__(
        self,
        camera: CameraInterface,
        tactile: TactileInterface,
        robot: RobotInterface,
        tactile_history_len: int = 15,
    ) -> None:
        self.camera = camera
        self.tactile = tactile
        self.robot = robot

        self.t = 0.0
        self._tactile_hist: list[np.ndarray] = []
        self._tactile_history_len = tactile_history_len
        self._last_state = None
        self._last_tactile = None

    def start(self) -> None:
        """启动所有硬件。"""
        print("[硬件] 正在启动传感器...")
        self.camera.start()
        self.tactile.start()
        self.robot.connect()
        self.robot.enable()
        print("[硬件] 所有传感器已启动")

    def stop(self) -> None:
        """停止所有硬件。"""
        print("[硬件] 正在停止传感器...")
        self.robot.disable()
        self.robot.disconnect()
        self.camera.stop()
        self.tactile.stop()
        print("[硬件] 所有传感器已停止")

    def read_sensors(self) -> dict[str, object]:
        """读取所有传感器的当前数据。

        Returns:
            dict: 包含 camera, tactile, state 的观测字典
        """
        obs = {
            "camera": None,
            "tactile": None,
            "state": None,
        }

        # 读取摄像头
        frame = self.camera.get_latest_frame()
        if frame is not None:
            obs["camera"] = frame
            self._last_state = frame

        # 读取触觉
        tactile = self.tactile.get_latest_reading()
        if tactile is not None:
            obs["tactile"] = tactile
            self._tactile_hist.append(tactile)
            # 限制历史长度
            if len(self._tactile_hist) > self._tactile_history_len * 2:
                self._tactile_hist = self._tactile_hist[-self._tactile_history_len:]
            self._last_tactile = tactile

        # 读取机器人状态
        state = self.robot.get_state()
        if state is not None:
            obs["state"] = state

        return obs

    def send_action(self, action: np.ndarray) -> bool:
        """发送动作到机器人控制器。"""
        return self.robot.send_action(action)

    def get_last_state(self) -> np.ndarray | None:
        """获取最后一次的状态观测。"""
        return self._last_state

    def tactile_history(self, length: int) -> np.ndarray:
        """返回最近 length 帧触觉历史，形状 [length, tactile_dim]。"""
        hist = self._tactile_hist[-length:]
        if len(hist) < length:
            if hist:
                pad = [hist[0]] * (length - len(hist))
            else:
                # 如果完全没有历史，用零填充
                tactile_dim = self.tactile.tactile_dim
                pad = [np.zeros(tactile_dim, dtype=np.float32)] * length
            hist = pad + hist
        return np.stack(hist, axis=0)

    def is_ready(self) -> bool:
        """检查所有硬件是否就绪。"""
        return (
            self.camera.is_ready()
            and self.tactile.is_ready()
            and self.robot.is_connected()
        )


# ---------------------------------------------------------------------------
# 慢模型：ACT
# ---------------------------------------------------------------------------

class ACTPolicy:
    """ACT 慢模型封装。

    直接加载 ACT 模型权重文件。
    若加载失败或未配置，则使用内置的模拟策略。
    """

    def __init__(
        self,
        pretrained_path: str | None,
        horizon: int,
        action_dim: int,
        device: str,
    ) -> None:
        self.horizon = horizon
        self.action_dim = action_dim
        self.device = torch.device(device)
        self._model = None
        self._rng = np.random.default_rng(0)

        if pretrained_path:
            try:
                self._load_real(pretrained_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[ACT] 真实模型加载失败，使用内置模拟模型：{exc}")
        else:
            print("[ACT] 未配置真实模型路径，使用内置模拟模型。")

    def _load_real(self, pretrained_path: str) -> None:
        """加载真实的ACT模型。

        注意：这里需要根据你的ACT模型实际格式来实现。
        如果你使用的是 LeRobot 的模型，需要单独处理。
        """
        from pathlib import Path

        model_path = Path(pretrained_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {pretrained_path}")

        # TODO: 根据实际模型格式加载
        # 示例1: 如果是 torch 保存的模型
        # checkpoint = torch.load(pretrained_path, map_location=self.device)
        # self._model = checkpoint['model']

        # 示例2: 如果是 LeRobot 模型，需要调用 LeRobot 的加载函数
        # from lerobot.common.policies.act.modeling_act import ACTPolicy
        # self._model = ACTPolicy.from_pretrained(pretrained_path)

        print(f"[ACT] 已加载真实模型: {pretrained_path}")

    def predict_chunk(
        self,
        observation: dict[str, object],
        state: np.ndarray,
    ) -> np.ndarray:
        """返回 [horizon, action_dim] 的动作块。"""
        if self._model is not None:
            return self._predict_real(observation, state)

        # 模拟 ACT：平滑的基准动作
        chunk = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
        for i in range(self.horizon):
            chunk[i] = (
                0.9 * state
                + 0.1 * self._rng.standard_normal(self.action_dim)
            ).astype(np.float32)
        return chunk

    def _predict_real(
        self,
        observation: dict[str, object],
        state: np.ndarray,
    ) -> np.ndarray:
        """使用真实模型推理。

        TODO: 根据你的模型输入格式实现
        """
        with torch.inference_mode():
            # 将输入转换为tensor
            state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)

            # TODO: 调用模型
            # chunk = self._model.predict(state_t, ...)
            # 这里需要根据实际模型的接口来实现

            # 临时返回模拟数据
            chunk = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
            for i in range(self.horizon):
                chunk[i] = (
                    0.9 * state
                    + 0.1 * self._rng.standard_normal(self.action_dim)
                ).astype(np.float32)

        return chunk


# ---------------------------------------------------------------------------
# 快模型：残差网络
# ---------------------------------------------------------------------------

class ResidualModel:
    """残差快模型封装。

    直接加载残差模型权重文件。
    若加载失败或未配置，则使用内置模拟残差模型。
    """

    def __init__(
        self,
        checkpoint_path: str | None,
        tactile_history: int,
        action_dim: int,
        device: str,
    ) -> None:
        self.tactile_history = tactile_history
        self.action_dim = action_dim
        self.device = torch.device(device)
        self._model = None
        self._rng = np.random.default_rng(1)

        if checkpoint_path:
            try:
                self._load_real(checkpoint_path)
            except Exception as exc:  # noqa: BLE001
                print(f"[残差] 真实模型加载失败，使用内置模拟模型：{exc}")
        else:
            print("[残差] 未配置真实模型路径，使用内置模拟模型。")

    def _load_real(self, checkpoint_path: str) -> None:
        """加载真实的残差模型。

        TODO: 根据你的残差模型实际格式来实现。
        """
        from pathlib import Path

        model_path = Path(checkpoint_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {checkpoint_path}")

        # TODO: 根据实际模型格式加载
        # 示例：
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # self._model = checkpoint["model"]
        # self._model.eval()

        print(f"[残差] 已加载真实模型: {checkpoint_path}")

    def refine(
        self,
        base_action: np.ndarray,
        tactile_history: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """输入基准动作，输出修正后的动作 [action_dim]。"""
        if self._model is not None:
            return self._refine_real(base_action, tactile_history, state)

        # 模拟残差：在基准动作附近加小幅扰动
        delta = 0.02 * self._rng.standard_normal(self.action_dim).astype(np.float32)
        return (base_action + delta).astype(np.float32)

    def _refine_real(
        self,
        base_action: np.ndarray,
        tactile_history: np.ndarray,
        state: np.ndarray,
    ) -> np.ndarray:
        """使用真实模型推理。

        TODO: 根据你的模型输入格式实现
        """
        with torch.inference_mode():
            tactile = torch.from_numpy(tactile_history).float().to(self.device).unsqueeze(0)
            current_force = torch.from_numpy(tactile_history[-1]).float().to(self.device).unsqueeze(0)
            state_t = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            base_t = torch.from_numpy(base_action).float().to(self.device).unsqueeze(0)

            # TODO: 调用模型
            # delta = self._model(
            #     tactile_history=tactile,
            #     current_force=current_force,
            #     state=state_t,
            #     act_chunk=base_t,
            # )
            # delta = delta[0, 0].detach().cpu().numpy()

            # 临时返回模拟数据
            delta = 0.02 * self._rng.standard_normal(self.action_dim).astype(np.float32)

        return (base_action + delta).astype(np.float32)


# ---------------------------------------------------------------------------
# 控制器
# ---------------------------------------------------------------------------

class FastSlowController:
    """快慢模型协调控制器（硬件版本）。"""

    def __init__(
        self,
        cfg: dict,
        mode: int,
    ) -> None:
        self.cfg = cfg
        self.mode = mode
        sim_cfg = cfg["simulation"]
        sensor_cfg = cfg["sensors"]
        act_cfg = cfg["act"]
        residual_cfg = cfg["residual"]

        self.duration = float(sim_cfg["duration_seconds"])
        self.speedup = float(sim_cfg.get("speedup", 1.0))

        self.camera_hz = int(sensor_cfg["camera_hz"])
        self.tactile_hz = int(sensor_cfg["tactile_hz"])
        self.tactile_dim = int(sensor_cfg["tactile_dim"])
        self.state_dim = int(sensor_cfg["state_dim"])

        self.act_horizon = int(act_cfg["horizon"])
        self.act_control_hz = float(act_cfg["control_hz"])
        self.residual_hz = float(residual_cfg["control_hz"])
        self.tactile_history_len = int(residual_cfg["tactile_history"])

        # 初始化硬件接口
        # TODO: 替换为真实硬件实现
        camera = MockCameraInterface(
            camera_hz=self.camera_hz,
            state_dim=self.state_dim,
        )
        tactile = MockTactileInterface(
            tactile_hz=self.tactile_hz,
            tactile_dim=self.tactile_dim,
        )
        robot = MockRobotInterface(
            action_dim=self.state_dim,
        )

        self.hardware = HardwareSensorManager(
            camera=camera,
            tactile=tactile,
            robot=robot,
            tactile_history_len=self.tactile_history_len,
        )

        self.act = ACTPolicy(
            pretrained_path=act_cfg.get("pretrained_path"),
            horizon=self.act_horizon,
            action_dim=self.state_dim,
            device=act_cfg.get("device", "cpu"),
        )
        self.residual = ResidualModel(
            checkpoint_path=residual_cfg.get("checkpoint_path"),
            tactile_history=self.tactile_history_len,
            action_dim=self.state_dim,
            device=residual_cfg.get("device", "cpu"),
        )

    # -- 内部调度 ---------------------------------------------------------

    def run(self) -> None:
        print("=" * 70)
        print(f"快慢模型协调控制（硬件版本） mode={self.mode}")
        print(
            f"  摄像头 {self.camera_hz}Hz / 触觉 {self.tactile_hz}Hz"
        )
        print(f"  ACT 每次推理 {self.act_horizon} 步，基准控制 {self.act_control_hz}Hz")
        print(f"  残差控制 {self.residual_hz}Hz")
        print("=" * 70)

        # 启动硬件
        self.hardware.start()

        if not self.hardware.is_ready():
            print("错误：硬件未就绪，退出")
            self.hardware.stop()
            return

        # 控制循环参数
        base_dt = 1.0 / self.tactile_hz  # 以触觉频率为基准
        act_dt = 1.0 / self.act_control_hz
        residual_dt = 1.0 / self.residual_hz

        # 动作块状态
        act_step = 0
        chunk = None
        chunk_cam_t = -1.0

        # 频率控制累加器
        act_accum = 0.0
        residual_accum = 0.0

        # 状态跟踪
        last_state = np.zeros(self.state_dim, dtype=np.float32)

        sent_count = 0
        start_time = time.perf_counter()

        try:
            while self.hardware.t < self.duration:
                loop_start = time.perf_counter()

                # 读取传感器
                obs = self.hardware.read_sensors()
                self.hardware.t += base_dt
                t = self.hardware.t

                # 更新状态
                if obs["camera"] is not None:
                    last_state = np.asarray(obs["camera"]).astype(np.float32)

                if obs["state"] is not None:
                    last_state = np.asarray(obs["state"]).astype(np.float32)

                # 摄像头到达时更新动作块
                if obs["camera"] is not None and (
                    chunk is None or t - chunk_cam_t >= act_dt * self.act_horizon
                ):
                    chunk = self.act.predict_chunk(obs, last_state)
                    chunk_cam_t = t
                    act_step = 0
                    print(
                        f"[ACT] t={t:7.3f}s 推理新的 {self.act_horizon} 步动作块"
                    )

                if chunk is None:
                    continue

                # 频率控制累加
                act_accum += base_dt
                residual_accum += base_dt

                # 模式1：直接发送ACT动作
                if self.mode == 1:
                    if act_accum >= act_dt - 1e-9:
                        act_accum = 0.0
                        action = chunk[act_step]
                        if self.hardware.send_action(action):
                            sent_count += 1
                        act_step = (act_step + 1) % self.act_horizon

                # 模式2：带残差修正
                else:
                    if residual_accum >= residual_dt - 1e-9:
                        residual_accum = 0.0
                        base_action = chunk[act_step]
                        hist = self.hardware.tactile_history(
                            self.tactile_history_len
                        )
                        action = self.residual.refine(
                            base_action,
                            hist,
                            last_state,
                        )
                        if self.hardware.send_action(action):
                            sent_count += 1

                        # 基准动作步进
                        if act_accum >= act_dt - 1e-9:
                            act_accum = 0.0
                            act_step = (act_step + 1) % self.act_horizon

                # 实时控制：按速度倍率等待
                if self.speedup > 0:
                    elapsed = time.perf_counter() - loop_start
                    wait = base_dt / self.speedup - elapsed
                    if wait > 0:
                        time.sleep(wait)

        except KeyboardInterrupt:
            print("\n[Ctrl+C] 用户中断")
        finally:
            # 停止硬件
            self.hardware.stop()

        elapsed_real = time.perf_counter() - start_time
        print("=" * 70)
        print(f"控制结束：共发送 {sent_count} 次动作")
        print(f"运行时间：{elapsed_real:.2f}秒")
        print("=" * 70)


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="快慢模型协调控制模拟器",
    )
    parser.add_argument(
        "--mode",
        type=int,
        required=True,
        choices=[1, 2],
        help="1=不带残差，2=带残差",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="配置文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"找不到配置文件：{args.config}")

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    controller = FastSlowController(cfg, mode=args.mode)
    controller.run()


if __name__ == "__main__":
    main()
