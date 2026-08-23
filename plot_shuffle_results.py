import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK SC"

conditions = [
    "normal",
    "shuffle_history",
    "zero_history",
    "noise_force_1.0",
    "noise_force_5.0",
    "noise_force_10.0",
]

full = {
    "normal": 22.787336,
    "shuffle_history": 22.787336,
    "zero_history": 22.787336,
    "noise_force_1.0": 28.042551,
    "noise_force_5.0": 31.136431,
    "noise_force_10.0": 31.782124,
}

no_t15 = {
    "normal": 7.639986,
    "shuffle_history": 7.639986,
    "zero_history": 7.639986,
    "noise_force_1.0": 8.622129,
    "noise_force_5.0": 12.638313,
    "noise_force_10.0": 13.944019,
}

labels = [
    "normal\n(基线)",
    "shuffle\nhistory",
    "zero\nhistory",
    "force噪声\n1.0",
    "force噪声\n5.0",
    "force噪声\n10.0",
]

x = np.arange(len(conditions))
width = 0.38

fig, axes = plt.subplots(1, 2, figsize=(14, 9.5))

for ax, data, title, n in (
    (axes[0], full, f"全量样本 (n=120680)", "full"),
    (axes[1], no_t15, f"排除token 15 (n=55247)", "no_t15"),
):
    vals = [data[c] for c in conditions]
    baseline = data["normal"]
    colors = [
        "#2e86ab",
        "#2e86ab",
        "#2e86ab",
        "#e07a5f",
        "#e07a5f",
        "#e07a5f",
    ]
    bars = ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.5)

    for bar, v in zip(bars, vals):
        change = (v / baseline - 1) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{v:.3f}\n({change:+.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MSE (pred_delta + act_chunk vs expert_action)", fontsize=10)
    ax.set_title(f"{title}", fontsize=12)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "触觉输入扰动实验 — 残差网络预测动作误差 (checkpoint_100.pth, full cache)",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=[0, 0.14, 1, 0.9])

conclusion = (
    "结论:\n"
    "1. 打乱/置0 history对输出零影响 (+0.00%)，VQ-VAE触觉历史分支实际无贡献;\n"
    "   排除token 15后结论不变，说明与码本分布无关。\n"
    "2. force噪声使误差单调上升，current_force是有效主导信息。\n"
    "3. 排除token 15后整体误差大幅下降 (22.79→7.64)，token 15(弱接触)样本最难预测。\n"
    "4. 非token 15样本对force更敏感 (噪声10.0: +82.5% vs 全量+39.5%)。"
)
fig.text(
    0.5,
    0.04,
    conclusion,
    ha="center",
    va="bottom",
    fontsize=10.5,
    linespacing=1.8,
    family="Noto Sans CJK SC",
    bbox=dict(boxstyle="round", facecolor="#f0f0f0", edgecolor="gray", pad=0.8),
)

out_path = "/home/yang/TactileEncoder/outputs/tactile_shuffle_experiment.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"chart saved: {out_path}")
