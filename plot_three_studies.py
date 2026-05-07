"""
3つの独立検証を1枚のグラフで並べるスクリプト。

- 本検証 (Llama-3.2-1B / GSM8K accuracy)
- 論文 (TinyLlama Single Phase / GSM8K accuracy)
- Zenn / seeda_yuto 氏 (OpenMythos / TinyStories perplexity → 逆数で擬似 accuracy 化)

3本を共通の y軸スケール（各実験のピーク値からの相対％）に正規化することで、
「動作の崖 → 早期飽和」という同一パターンが3独立検証で観測されたことを可視化。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent

# === 本検証 Llama-3.2-1B GSM8K accuracy ===
# results/gsm8k_num_steps_*.json から自動取得
local = {}
for p in sorted((OUT_DIR / "results").glob("gsm8k_num_steps_*.json")):
    d = json.load(open(p))
    local[d["num_steps"]] = d["accuracy"] * 100  # %
local_xs = sorted(local.keys())
local_ys = [local[x] for x in local_xs]

# === 論文 Appendix Table 6: TinyLlama 4,8,4 (Train Recurrence=4) Single Phase, GSM8K ===
paper_xs = [1, 2, 4, 8, 16, 32]
paper_ys = [18.9, 29.3, 36.4, 40.2, 40.3, 40.3]

# === 論文 Appendix Table 6: TinyLlama 4,8,4 (Train Recurrence=4) Two Phase, GSM8K ===
# 表抽出より recurrence 1 と 32 が確実 (23.4, 49.3)。中間値は本文では不可視だったが、
# 図 5 のシェイプから本検証と同様 8 ループでほぼ飽和と読める。
# Two Phase のカーブも参考として加える。
paper2_xs = [1, 32]
paper2_ys = [23.4, 49.3]

# === Zenn / seeda_yuto 氏 OpenMythos TinyStories Perplexity (large, 0.042B) ===
# perplexity は小さいほど良いので、逆数 (1/PPL) を取り「擬似 accuracy 比」にする
zenn_xs = [1, 2, 4, 8, 16]
zenn_ppl = [169.0, 7.1, 7.0, 6.9, 6.9]  # 1回: 完全崩壊、2回以降: ほぼフラット
zenn_inv = [1.0 / p for p in zenn_ppl]

# 3本を「各実験のピーク値からの相対％」に正規化
def normalize(ys):
    peak = max(ys)
    return [y / peak * 100 for y in ys]

local_norm = normalize(local_ys)
paper_norm = normalize(paper_ys)
zenn_norm = normalize(zenn_inv)


# === グラフ A: 絶対値プロット（3つの subplot） ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# 左: 本検証
axes[0].plot(local_xs, local_ys, "o-", color="C0", linewidth=2, markersize=8)
for x, y in zip(local_xs, local_ys):
    axes[0].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
axes[0].set_xlabel("num_steps")
axes[0].set_ylabel("GSM8K accuracy (%)")
axes[0].set_title("(A) This study\nRetrofitted Llama-3.2-1B")
axes[0].set_xticks(local_xs)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 60)

# 中央: 論文 TinyLlama Single Phase
axes[1].plot(paper_xs, paper_ys, "s-", color="C2", linewidth=2, markersize=8, label="Single Phase")
axes[1].plot(paper2_xs, paper2_ys, "x--", color="C3", linewidth=2, markersize=10, label="Two Phase (n=1, 32 only)")
for x, y in zip(paper_xs, paper_ys):
    axes[1].annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(paper_xs)
axes[1].set_xticklabels(paper_xs)
axes[1].set_xlabel("recurrence (log scale)")
axes[1].set_ylabel("GSM8K accuracy (%)")
axes[1].set_title("(B) Paper arXiv:2511.07384\nTinyLlama-1.1B (Table 6)")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 60)
axes[1].legend(loc="lower right", fontsize=9)

# 右: Zenn TinyStories Perplexity
ax2 = axes[2]
ax2.plot(zenn_xs, zenn_ppl, "^-", color="C4", linewidth=2, markersize=8)
for x, y in zip(zenn_xs, zenn_ppl):
    ax2.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax2.set_xscale("log", base=2)
ax2.set_xticks(zenn_xs)
ax2.set_xticklabels(zenn_xs)
ax2.set_yscale("log")
ax2.set_xlabel("loop count (log scale)")
ax2.set_ylabel("Perplexity (log, lower = better)")
ax2.set_title("(C) Zenn / seeda_yuto\nOpenMythos large + TinyStories")
ax2.grid(True, alpha=0.3, which="both")

plt.suptitle("Three independent studies: cliff at loop 1→2, then early saturation",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_three_studies.png", dpi=120, bbox_inches="tight")
plt.close()
print(f"saved: plot_three_studies.png")


# === グラフ B: 正規化プロット（3本を1枚に重ねる） ===
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(local_xs, local_norm, "o-", color="C0", linewidth=2.5, markersize=10,
        label=f"This study (Llama-3.2-1B, GSM8K, peak={max(local_ys):.1f}%)")
ax.plot(paper_xs, paper_norm, "s-", color="C2", linewidth=2.5, markersize=10,
        label=f"Paper TinyLlama Single Phase (peak={max(paper_ys):.1f}%)")
ax.plot(zenn_xs, zenn_norm, "^-", color="C4", linewidth=2.5, markersize=10,
        label=f"Zenn TinyStories perplexity (1/PPL, peak={max(zenn_inv):.4f})")

# 動作の崖領域をマーキング
ax.axvspan(1, 2, alpha=0.15, color="orange", label="Cliff zone (1→2)")
ax.axhline(y=90, color="gray", linestyle=":", alpha=0.6)
ax.text(28, 91, "90% of peak", fontsize=9, color="gray")

ax.set_xscale("log", base=2)
all_xs = sorted(set(local_xs) | set(paper_xs) | set(zenn_xs))
ax.set_xticks(all_xs)
ax.set_xticklabels(all_xs)
ax.set_xlabel("loop count / num_steps (log scale)", fontsize=12)
ax.set_ylabel("Performance relative to each study's peak (%)", fontsize=12)
ax.set_title("Same pattern across 3 independent studies:\ncliff at 1→2, then early saturation by loop ≈ 4-8",
             fontsize=12)
ax.grid(True, alpha=0.3, which="both")
ax.legend(loc="lower right", fontsize=10)
ax.set_ylim(-10, 110)

plt.tight_layout()
plt.savefig(OUT_DIR / "plot_three_studies_normalized.png", dpi=120)
plt.close()
print(f"saved: plot_three_studies_normalized.png")
