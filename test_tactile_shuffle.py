import torch
from torch.utils.data import DataLoader, Dataset

from load_model import load_model_checkpoint
from model import VQVAEForceEncoder

CACHE_PATH = "/home/yang/TactileEncoder/outputs/act_cache/act_cache_val.pt"
CHECKPOINT_PATH = "/home/yang/TactileEncoder/residual_actmodel/checkpoints_novqvae/checkpoint_200.pth"
VQVAE_CKPT_PATH = "/home/yang/TactileEncoder/TactileSelfencoder/vqvae_checkpoints/decoder_mlp/16tokens/checkpoint_epoch_200.pth"
BATCH_SIZE = 512
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"


class CacheDataset(Dataset):
    def __init__(self, cache):
        self.keys = sorted(cache.keys())
        self.cache = cache

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        entry = self.cache[self.keys[idx]]
        return {
            "tactile_history": entry["tactile_history"].float(),
            "current_force": entry["current_force"].float(),
            "state": entry["state"].float(),
            "act_chunk": entry["act_chunk"].float(),
            "expert_action": entry["expert_action"].float(),
            "act_visual_tokens": entry["act_visual"].float(),
        }


def collate(batch):
    out = {}
    for key in batch[0]:
        out[key] = torch.stack([b[key] for b in batch])
    return out


UNITS_PER_DEGREE = 11.37  # 11.37单位 = 1度


@torch.no_grad()
def analyze_delta_axes(model, loader, vqvae_encoder, target_token=None):
    """统计pred_delta每个动作轴的平均修正量（转换为角度）。

    target_token: 若指定(如15)，只统计该token的样本；None则统计全部。
    返回 (axes_stats, n), axes_stats[i] = dict(mean, abs_mean, max) 单位=度。
    """
    axes_stats = [
        {"mean_sum": 0.0, "abs_sum": 0.0, "max": 0.0}
        for _ in range(6)
    ]
    n = 0

    for batch in loader:
        B = batch["tactile_history"].shape[0]
        batch = {k: v.to(device) for k, v in batch.items()}

        if target_token is not None:
            hist = batch["tactile_history"]
            with torch.no_grad():
                _, tid = vqvae_encoder(hist, return_token_id=True)
            mask = tid == target_token
            if not mask.any():
                continue
            batch = {k: v[mask] for k, v in batch.items()}
            B = batch["tactile_history"].shape[0]

        pred_delta, _ = model(
            batch["tactile_history"],
            batch["current_force"],
            batch["state"],
            batch["act_chunk"],
            act_visual_tokens=batch.get("act_visual_tokens"),
            return_feature_metrics=True,
        )
        delta = pred_delta.float() / UNITS_PER_DEGREE  # [B, T, 6] 转为度
        for d in range(6):
            col = delta[:, :, d]
            axes_stats[d]["mean_sum"] += col.mean().item() * B
            axes_stats[d]["abs_sum"] += col.abs().mean().item() * B
            axes_stats[d]["max"] = max(axes_stats[d]["max"], col.abs().max().item())
        n += B

    return axes_stats, n


def print_delta_axis_report(title, axes_stats, n):
    print(f"\n{title} (n={n}, 单位=度, 1度={UNITS_PER_DEGREE}单位)")
    print("-" * 70)
    print(f"{'轴':<6}{'均值mean':>12}{'绝对值均值abs_mean':>18}{'绝对最大值abs_max':>16}")
    print("-" * 70)
    for d, s in enumerate(axes_stats):
        mean = s["mean_sum"] / max(n, 1)
        abs_mean = s["abs_sum"] / max(n, 1)
        print(f"dim {d:<2}{mean:>14.6f}{abs_mean:>18.6f}{s['max']:>16.6f}")
    print("-" * 70)


@torch.no_grad()
def evaluate(model, loader, mode, generator):
    """返回每个样本的 MSE(pred_delta + act_chunk, expert_action)。"""
    mses = []

    for batch in loader:
        B = batch["tactile_history"].shape[0]
        batch = {k: v.to(device) for k, v in batch.items()}
        expert = batch["expert_action"]

        if mode == "normal":
            pass
        elif mode == "act_baseline":
            # ACT原始基线：直接使用act_chunk作为预测，不经过residual模型
            pred_action = batch["act_chunk"]
            mse = (pred_action - expert).square().mean(dim=(1, 2))
            mses.append(mse.cpu())
            continue
        elif mode == "shuffle_history":
            perm = torch.randperm(B, generator=generator)
            batch["tactile_history"] = batch["tactile_history"][perm]
        elif mode == "zero_history":
            batch["tactile_history"] = torch.zeros_like(batch["tactile_history"])
        elif mode == "zero_force":
            batch["current_force"] = torch.zeros_like(batch["current_force"])
        elif mode.startswith("noise_force_"):
            sigma = float(mode.split("_")[-1])
            noise = torch.randn_like(batch["current_force"]) * sigma
            batch["current_force"] = batch["current_force"] + noise
        elif mode == "shuffle_visual":
            perm = torch.randperm(B, generator=generator)
            batch["act_visual_tokens"] = batch["act_visual_tokens"][perm]
        elif mode == "zero_visual":
            batch["act_visual_tokens"] = torch.zeros_like(batch["act_visual_tokens"])
        elif mode.startswith("noise_visual_"):
            sigma = float(mode.split("_")[-1])
            noise = torch.randn_like(batch["act_visual_tokens"]) * sigma
            batch["act_visual_tokens"] = batch["act_visual_tokens"] + noise
        else:
            raise ValueError(mode)

        pred_delta, _ = model(
            batch["tactile_history"],
            batch["current_force"],
            batch["state"],
            batch["act_chunk"],
            act_visual_tokens=batch.get("act_visual_tokens"),
            return_feature_metrics=True,
        )
        pred_action = pred_delta + batch["act_chunk"]
        mse = (pred_action - expert).square().mean(dim=(1, 2))
        mses.append(mse.cpu())

    return torch.cat(mses), float(torch.cat(mses).mean().item())


def compute_token_ids(loader, batch_size, vqvae_encoder=None):
    """返回所有样本的VQ-VAE token ID (按dataset顺序)。

    优先使用传入的独立vqvae_encoder；否则回退到model.tactile_encoder。
    """
    token_ids = []
    for i in range(0, len(loader.dataset), batch_size):
        keys = loader.dataset.keys[i:i + batch_size]
        hist = torch.stack([loader.dataset.cache[k]["tactile_history"] for k in keys]).float().to(device)
        with torch.no_grad():
            _, tid = vqvae_encoder(hist, return_token_id=True)
        token_ids.extend(tid.cpu().tolist())
    return token_ids


class SubsetCacheDataset(CacheDataset):
    def __init__(self, cache, indices):
        super().__init__(cache)
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return super().__getitem__(self.indices[idx])


def main():
    torch.manual_seed(SEED)
    model = load_model_checkpoint(CHECKPOINT_PATH, device=device)["model"]

    # 独立VQ-VAE编码器（用于计算token ID，即使residual模型已去掉vqvae分支）
    vqvae_encoder = VQVAEForceEncoder(
        vqvae_checkpoint_path=VQVAE_CKPT_PATH,
    ).to(device)
    vqvae_encoder.eval()

    cache = torch.load(CACHE_PATH, map_location="cpu")
    print(f"总样本数: {len(cache)}")

    full_loader = DataLoader(
        CacheDataset(cache),
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
    )

    g = torch.Generator().manual_seed(SEED)

    # 模型是否包含触觉历史分支
    use_tactile = hasattr(model, "tactile_encoder")
    print(f"模型含触觉历史分支: {use_tactile}")

    # ===== 全量实验 =====
    mse_act_baseline, mean_act_baseline = evaluate(model, full_loader, "act_baseline", g)
    mse_normal, mean_normal = evaluate(model, full_loader, "normal", g)
    mse_zf, mean_zf = evaluate(model, full_loader, "zero_force", g)
    mse_n1, mean_n1 = evaluate(model, full_loader, "noise_force_1.0", g)
    mse_n5, mean_n5 = evaluate(model, full_loader, "noise_force_5.0", g)
    mse_n10, mean_n10 = evaluate(model, full_loader, "noise_force_10.0", g)
    mse_vs, mean_vs = evaluate(model, full_loader, "shuffle_visual", g)
    mse_vz, mean_vz = evaluate(model, full_loader, "zero_visual", g)
    mse_v1, mean_v1 = evaluate(model, full_loader, "noise_visual_0.1", g)
    mse_v5, mean_v5 = evaluate(model, full_loader, "noise_visual_0.5", g)
    mse_v10, mean_v10 = evaluate(model, full_loader, "noise_visual_1.0", g)
    if use_tactile:
        mse_hist, mean_hist = evaluate(model, full_loader, "shuffle_history", g)
        mse_zh, mean_zh = evaluate(model, full_loader, "zero_history", g)

    print("=" * 60)
    print("触觉输入实验 (MSE: pred_action vs expert_action)")
    print("=" * 60)
    print(f"act_baseline       : {mean_act_baseline:.6f}")
    print(f"normal (residual)  : {mean_normal:.6f}")
    if use_tactile:
        print(f"shuffle_history    : {mean_hist:.6f}")
        print(f"zero_history       : {mean_zh:.6f}")
    print(f"zero_force         : {mean_zf:.6f}")
    print(f"noise_force_1.0    : {mean_n1:.6f}")
    print(f"noise_force_5.0    : {mean_n5:.6f}")
    print(f"noise_force_10.0   : {mean_n10:.6f}")
    print(f"shuffle_visual     : {mean_vs:.6f}")
    print(f"zero_visual        : {mean_vz:.6f}")
    print(f"noise_visual_0.1   : {mean_v1:.6f}")
    print(f"noise_visual_0.5   : {mean_v5:.6f}")
    print(f"noise_visual_1.0   : {mean_v10:.6f}")
    print("-" * 60)

    if use_tactile:
        improved_hist = (mse_hist < mse_normal).float().mean().item() * 100
        improved_zh = (mse_zh < mse_normal).float().mean().item() * 100
        print(f"打乱history后更接近GT占比: {improved_hist:.2f}%")
        print(f"置0 history后更接近GT占比: {improved_zh:.2f}%")
        print("-" * 60)
        print(f"打乱history:         平均误差变化 {mean_hist / mean_normal * 100 - 100:+.2f}%")
        print(f"置0 history:         平均误差变化 {mean_zh / mean_normal * 100 - 100:+.2f}%")
    print(f"ACT基线 → Residual:  平均误差变化 {mean_normal / mean_act_baseline * 100 - 100:+.2f}%")
    print(f"置0 force:           平均误差变化 {mean_zf / mean_normal * 100 - 100:+.2f}%")
    print(f"force噪声1.0:        平均误差变化 {mean_n1 / mean_normal * 100 - 100:+.2f}%")
    print(f"force噪声5.0:        平均误差变化 {mean_n5 / mean_normal * 100 - 100:+.2f}%")
    print(f"force噪声10.0:       平均误差变化 {mean_n10 / mean_normal * 100 - 100:+.2f}%")
    print(f"打乱visual:         平均误差变化 {mean_vs / mean_normal * 100 - 100:+.2f}%")
    print(f"置0 visual:         平均误差变化 {mean_vz / mean_normal * 100 - 100:+.2f}%")
    print(f"visual噪声0.1:      平均误差变化 {mean_v1 / mean_normal * 100 - 100:+.2f}%")
    print(f"visual噪声0.5:      平均误差变化 {mean_v5 / mean_normal * 100 - 100:+.2f}%")
    print(f"visual噪声1.0:      平均误差变化 {mean_v10 / mean_normal * 100 - 100:+.2f}%")
    print("=" * 60)

    # ===== 排除 token 15 的实验 =====
    print("\n计算VQ-VAE token ID (排除token 15)...")
    token_ids = compute_token_ids(full_loader, BATCH_SIZE, vqvae_encoder=vqvae_encoder)
    keep_indices = [
        i for i, tid in enumerate(token_ids) if tid != 15
    ]
    print(f"排除token 15后剩余样本数: {len(keep_indices)} / {len(token_ids)}")

    subset_loader = DataLoader(
        SubsetCacheDataset(cache, keep_indices),
        batch_size=BATCH_SIZE,
        num_workers=4,
        shuffle=False,
    )

    mse_sub_act_baseline, mean_sub_act_baseline = evaluate(model, subset_loader, "act_baseline", g)
    mse_sub_normal, mean_sub_normal = evaluate(model, subset_loader, "normal", g)
    mse_sub_zf, mean_sub_zf = evaluate(model, subset_loader, "zero_force", g)
    mse_sub_n1, mean_sub_n1 = evaluate(model, subset_loader, "noise_force_1.0", g)
    mse_sub_n5, mean_sub_n5 = evaluate(model, subset_loader, "noise_force_5.0", g)
    mse_sub_n10, mean_sub_n10 = evaluate(model, subset_loader, "noise_force_10.0", g)
    mse_sub_vs, mean_sub_vs = evaluate(model, subset_loader, "shuffle_visual", g)
    mse_sub_vz, mean_sub_vz = evaluate(model, subset_loader, "zero_visual", g)
    mse_sub_v1, mean_sub_v1 = evaluate(model, subset_loader, "noise_visual_0.1", g)
    mse_sub_v5, mean_sub_v5 = evaluate(model, subset_loader, "noise_visual_0.5", g)
    mse_sub_v10, mean_sub_v10 = evaluate(model, subset_loader, "noise_visual_1.0", g)
    if use_tactile:
        mse_sub_hist, mean_sub_hist = evaluate(model, subset_loader, "shuffle_history", g)
        mse_sub_zh, mean_sub_zh = evaluate(model, subset_loader, "zero_history", g)

    print("=" * 60)
    print(f"排除token 15实验 (n={len(keep_indices)})")
    print("=" * 60)
    print(f"act_baseline       : {mean_sub_act_baseline:.6f}")
    print(f"normal (residual)  : {mean_sub_normal:.6f}")
    if use_tactile:
        print(f"shuffle_history    : {mean_sub_hist:.6f}")
        print(f"zero_history       : {mean_sub_zh:.6f}")
    print(f"zero_force         : {mean_sub_zf:.6f}")
    print(f"noise_force_1.0    : {mean_sub_n1:.6f}")
    print(f"noise_force_5.0    : {mean_sub_n5:.6f}")
    print(f"noise_force_10.0   : {mean_sub_n10:.6f}")
    print(f"shuffle_visual     : {mean_sub_vs:.6f}")
    print(f"zero_visual        : {mean_sub_vz:.6f}")
    print(f"noise_visual_0.1   : {mean_sub_v1:.6f}")
    print(f"noise_visual_0.5   : {mean_sub_v5:.6f}")
    print(f"noise_visual_1.0   : {mean_sub_v10:.6f}")
    print("-" * 60)

    if use_tactile:
        improved_hist = (mse_sub_hist < mse_sub_normal).float().mean().item() * 100
        improved_zh = (mse_sub_zh < mse_sub_normal).float().mean().item() * 100
        print(f"打乱history后更接近GT占比: {improved_hist:.2f}%")
        print(f"置0 history后更接近GT占比: {improved_zh:.2f}%")
        print("-" * 60)
        print(f"打乱history:         平均误差变化 {mean_sub_hist / mean_sub_normal * 100 - 100:+.2f}%")
        print(f"置0 history:         平均误差变化 {mean_sub_zh / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"ACT基线 → Residual:  平均误差变化 {mean_sub_normal / mean_sub_act_baseline * 100 - 100:+.2f}%")
    print(f"置0 force:           平均误差变化 {mean_sub_zf / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"force噪声1.0:        平均误差变化 {mean_sub_n1 / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"force噪声5.0:        平均误差变化 {mean_sub_n5 / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"force噪声10.0:       平均误差变化 {mean_sub_n10 / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"打乱visual:         平均误差变化 {mean_sub_vs / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"置0 visual:         平均误差变化 {mean_sub_vz / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"visual噪声0.1:      平均误差变化 {mean_sub_v1 / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"visual噪声0.5:      平均误差变化 {mean_sub_v5 / mean_sub_normal * 100 - 100:+.2f}%")
    print(f"visual噪声1.0:      平均误差变化 {mean_sub_v10 / mean_sub_normal * 100 - 100:+.2f}%")
    print("=" * 60)

    # ===== token 15 pred_delta 每个轴修正统计 =====
    print("\n计算pred_delta每个动作轴的修正统计...")
    axes_all, n_all = analyze_delta_axes(model, full_loader, vqvae_encoder)
    print_delta_axis_report("全量样本 pred_delta 每轴修正", axes_all, n_all)
    axes_t15, n_t15 = analyze_delta_axes(
        model, full_loader, vqvae_encoder, target_token=15
    )
    print_delta_axis_report("token 15 样本 pred_delta 每轴修正", axes_t15, n_t15)

    del cache
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""
如需重新生成完整缓存（含 tactile_history/current_force/state/expert_action），
从项目根目录运行:

    python -m dataloader.preprocess_act_cache \
        --config dataloader/tactile_dataloader.yaml \
        --output outputs/act_cache/act_cache_val.pt \
        --full-cache
"""