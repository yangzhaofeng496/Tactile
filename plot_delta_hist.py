import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from load_model import load_model_checkpoint
from model import VQVAEForceEncoder

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK SC"

CACHE_PATH = "/home/yang/TactileEncoder/outputs/act_cache/act_cache.pt"
CHECKPOINT_PATH = "/home/yang/TactileEncoder/residual_actmodel/checkpoints_novqvae/checkpoint_100.pth"
VQVAE_CKPT_PATH = "/home/yang/TactileEncoder/TactileSelfencoder/vqvae_checkpoints/decoder_mlp/16tokens/checkpoint_epoch_200.pth"
BATCH_SIZE = 512
UNITS_PER_DEGREE = 11.37

device = "cuda" if torch.cuda.is_available() else "cpu"


def collect_delta(model, loader, vqvae_encoder, target_token=None):
    """收集每个轴的pred_delta值(单位=度)。返回list of 6 个numpy数组。"""
    collected = [[] for _ in range(6)]
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if target_token is not None:
                _, tid = vqvae_encoder(batch["tactile_history"], return_token_id=True)
                mask = tid == target_token
                if not mask.any():
                    continue
                batch = {k: v[mask] for k, v in batch.items()}
            pred_delta, _ = model(
                batch["tactile_history"],
                batch["current_force"],
                batch["state"],
                batch["act_chunk"],
                act_visual_tokens=batch.get("act_visual_tokens"),
                return_feature_metrics=True,
            )
            delta = (pred_delta.float() / UNITS_PER_DEGREE).cpu().numpy()  # [B,30,6] 度
            for d in range(6):
                collected[d].append(delta[:, :, d].ravel())
            n += delta.shape[0]
    return [np.concatenate(c) for c in collected], n


def main():
    torch.manual_seed(42)
    model = load_model_checkpoint(CHECKPOINT_PATH, device=device)["model"]
    vqvae_encoder = VQVAEForceEncoder(vqvae_checkpoint_path=VQVAE_CKPT_PATH).to(device)
    vqvae_encoder.eval()

    cache = torch.load(CACHE_PATH, map_location="cpu")

    class CacheDataset(torch.utils.data.Dataset):
        def __init__(self, cache):
            self.keys = sorted(cache.keys())
            self.cache = cache

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, idx):
            e = self.cache[self.keys[idx]]
            return {
                "tactile_history": e["tactile_history"].float(),
                "current_force": e["current_force"].float(),
                "state": e["state"].float(),
                "act_chunk": e["act_chunk"].float(),
                "expert_action": e["expert_action"].float(),
                "act_visual_tokens": e["act_visual"].float(),
            }

    def collate(batch):
        return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

    full_loader = DataLoader(
        CacheDataset(cache), batch_size=BATCH_SIZE, num_workers=4, shuffle=False,
        collate_fn=collate,
    )

    print("收集全量样本...")
    all_axes, n_all = collect_delta(model, full_loader, vqvae_encoder)
    print("收集token 15样本...")
    t15_axes, n_t15 = collect_delta(model, full_loader, vqvae_encoder, target_token=15)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    bins = np.linspace(-20, 20, 121)  # 0.33度/箱

    axis_names = ["dim 0", "dim 1", "dim 2", "dim 3", "dim 4", "dim 5"]
    for d in range(6):
        ax = axes[d // 3][d % 3]
        ax.hist(
            all_axes[d], bins=bins, alpha=0.6, color="#2e86ab",
            label=f"全量 (n={n_all})",
        )
        ax.hist(
            t15_axes[d], bins=bins, alpha=0.6, color="#e07a5f",
            label=f"token 15 (n={n_t15})",
        )
        ax.set_title(f"{axis_names[d]}  pred_delta修正分布", fontsize=12)
        ax.set_xlabel("修正量 (度)")
        ax.set_ylabel("次数")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        "pred_delta 每轴修正量分布 (单位:度, 1度=11.37单位)  checkpoint_100.pth",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "/home/yang/TactileEncoder/outputs/delta_axis_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"chart saved: {out}")

    # 顺便打印每轴均值和绝对值均值
    print(f"\n{'轴':<8}{'全量mean°':>12}{'全量abs°':>12}{'t15 mean°':>12}{'t15 abs°':>12}")
    for d in range(6):
        print(
            f"dim{d:<4}{all_axes[d].mean():>12.4f}{np.abs(all_axes[d]).mean():>12.4f}"
            f"{t15_axes[d].mean():>12.4f}{np.abs(t15_axes[d]).mean():>12.4f}"
        )


if __name__ == "__main__":
    main()
