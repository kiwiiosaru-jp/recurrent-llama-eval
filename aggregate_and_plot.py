"""
Phase 4: 結果ファイルを読み込んでグラフ化・レポート生成。

使い方:
  python aggregate_and_plot.py            # ローカル results/ を集計
  python aggregate_and_plot.py results_colab/results  # Colab結果も同じ形式で
  python aggregate_and_plot.py --compare  # ローカルとColabの両方を比較プロット
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt


def load_results(results_dir: Path):
    """results_dir/gsm8k_num_steps_*.json を全部読む。順序付き。"""
    out = {}
    for p in sorted(results_dir.glob("gsm8k_num_steps_*.json")):
        with open(p) as f:
            d = json.load(f)
        out[d["num_steps"]] = d
    return out


def make_plot(local: dict, colab: dict | None, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # 左: accuracy vs num_steps
    if local:
        xs = sorted(local.keys())
        ys = [local[x]["accuracy"] * 100 for x in xs]
        ax1.plot(xs, ys, "o-", color="C0", linewidth=2, markersize=10, label="Local (M2 Pro)")
        for x, y in zip(xs, ys):
            ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=10, color="C0")
    if colab:
        xs = sorted(colab.keys())
        ys = [colab[x]["accuracy"] * 100 for x in xs]
        ax1.plot(xs, ys, "s--", color="C1", linewidth=2, markersize=10, label="Colab (T4)")
        for x, y in zip(xs, ys):
            ax1.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                         xytext=(0, -16), ha="center", fontsize=10, color="C1")
    ax1.set_xlabel("num_steps (inference recurrence)", fontsize=12)
    ax1.set_ylabel("Accuracy on GSM8K (%)", fontsize=12)
    ax1.set_title("Recurrent-Llama-3.2: recurrence vs accuracy (n=100)", fontsize=12)
    ax1.set_xticks(sorted(set(local.keys()) | (set(colab.keys()) if colab else set())))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # 右: 推論時間
    if local:
        xs = sorted(local.keys())
        ys = [local[x]["elapsed_sec"] for x in xs]
        ax2.bar([x - 0.2 for x in xs], ys, width=0.4, color="C0", label="Local (M2 Pro)")
    if colab:
        xs = sorted(colab.keys())
        ys = [colab[x]["elapsed_sec"] for x in xs]
        ax2.bar([x + 0.2 for x in xs], ys, width=0.4, color="C1", label="Colab (T4)")
    ax2.set_xlabel("num_steps", fontsize=12)
    ax2.set_ylabel("Inference time for 100 problems (sec)", fontsize=12)
    ax2.set_title("Inference time", fontsize=12)
    ax2.set_xticks(sorted(set(local.keys()) | (set(colab.keys()) if colab else set())))
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved plot: {out_path}")


def make_marginal_plot(local: dict, colab: dict | None, out_path: Path):
    """差分（前のループからの伸び）プロット。"""
    fig, ax = plt.subplots(figsize=(9, 5))

    def _marginals(d):
        xs = sorted(d.keys())
        accs = [d[x]["accuracy"] * 100 for x in xs]
        diffs = [accs[i] - accs[i-1] for i in range(1, len(accs))]
        labels = [f"{xs[i-1]}→{xs[i]}" for i in range(1, len(xs))]
        return labels, diffs

    if local:
        labels, diffs = _marginals(local)
        x_pos = list(range(len(diffs)))
        bars = ax.bar([p - 0.2 for p in x_pos], diffs, width=0.4, color="C0", label="Local (M2 Pro)")
        for p, d in zip([pp - 0.2 for pp in x_pos], diffs):
            ax.text(p, d + (0.5 if d >= 0 else -1.5), f"{d:+.0f}", ha="center", fontsize=10)

    if colab:
        labels_c, diffs_c = _marginals(colab)
        x_pos = list(range(len(diffs_c)))
        ax.bar([p + 0.2 for p in x_pos], diffs_c, width=0.4, color="C1", label="Colab (T4)")
        for p, d in zip([pp + 0.2 for pp in x_pos], diffs_c):
            ax.text(p, d + (0.5 if d >= 0 else -1.5), f"{d:+.0f}", ha="center", fontsize=10, color="C1")

    ax.axhline(0, color="black", linewidth=0.8)
    if local:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
    elif colab:
        ax.set_xticks(range(len(labels_c)))
        ax.set_xticklabels(labels_c)
    ax.set_xlabel("transition (num_steps)", fontsize=12)
    ax.set_ylabel("Marginal accuracy gain (pt)", fontsize=12)
    ax.set_title("Marginal improvement per loop", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"saved marginal plot: {out_path}")


def make_report(local: dict, colab: dict | None, out_path: Path):
    lines = []
    lines.append("# Recurrent-Llama-3.2 GSM8K 100問評価レポート")
    lines.append("")
    lines.append(f"**モデル**: smcleish/Recurrent-Llama-3.2-train-recurrence-16  ")
    lines.append(f"**評価データ**: GSM8K test 先頭100問  ")
    lines.append(f"**プロンプト**: 4-shot CoT (lm-eval-harness 風)  ")
    lines.append(f"**生成設定**: greedy, max_new_tokens=160, stop on '\\nQuestion:'")
    lines.append("")
    lines.append("## 結果サマリ")
    lines.append("")
    lines.append("### ローカル (M2 Pro / MPS / fp16)")
    if local:
        lines.append("")
        lines.append("| num_steps | accuracy | 正答数 | 経過時間 |")
        lines.append("|---:|---:|---:|---:|")
        for ns in sorted(local.keys()):
            d = local[ns]
            lines.append(f"| {ns} | {d['accuracy']*100:.1f}% | {d['n_correct']}/{d['n_samples']} | {d['elapsed_sec']:.1f}s |")
    else:
        lines.append("（結果なし）")
    lines.append("")

    if colab:
        lines.append("### Colab (T4 / fp16)")
        lines.append("")
        lines.append("| num_steps | accuracy | 正答数 | 経過時間 |")
        lines.append("|---:|---:|---:|---:|")
        for ns in sorted(colab.keys()):
            d = colab[ns]
            lines.append(f"| {ns} | {d['accuracy']*100:.1f}% | {d['n_correct']}/{d['n_samples']} | {d['elapsed_sec']:.1f}s |")
        lines.append("")

    lines.append("## 観察")
    lines.append("")
    src = local if local else (colab or {})
    if len(src) >= 2:
        keys = sorted(src.keys())
        a0 = src[keys[0]]["accuracy"]
        a1 = src[keys[-1]]["accuracy"]
        diff = (a1 - a0) * 100
        lines.append(f"- num_steps={keys[0]} → {keys[-1]} で accuracy が {a0*100:.1f}% → {a1*100:.1f}% ({diff:+.1f}pt)")
        if diff > 5:
            lines.append("- **論文の主張通り、ループ増で精度が向上する傾向**")
        elif diff > 0:
            lines.append("- 緩やかな改善傾向")
        else:
            lines.append("- 改善なし／むしろ低下（小サンプル100問のノイズの可能性）")

    # 正誤例の抜粋（local 優先、なければ colab）
    src_max = None
    src_min = None
    if local:
        ks = sorted(local.keys())
        src_min, src_max = local[ks[0]], local[ks[-1]]
    elif colab:
        ks = sorted(colab.keys())
        src_min, src_max = colab[ks[0]], colab[ks[-1]]

    if src_max is not None and src_min is not None:
        lines.append("")
        lines.append("## 「最少ループでは外したが最多ループで正解した」例")
        lines.append("")
        flipped = []
        for r_max in src_max["records"]:
            r_min = src_min["records"][r_max["idx"]]
            if r_max["correct"] and not r_min["correct"]:
                flipped.append((r_min, r_max))
        if not flipped:
            lines.append("（該当なし）")
        else:
            for r_min, r_max in flipped[:3]:
                lines.append(f"### 問題 #{r_min['idx']}")
                lines.append("")
                lines.append(f"**Q**: {r_min['question']}")
                lines.append("")
                lines.append(f"**正解**: {r_min['gold']}")
                lines.append("")
                lines.append(f"- num_steps={src_min['num_steps']} の予測: `{r_min['pred']}` ❌")
                lines.append(f"- num_steps={src_max['num_steps']} の予測: `{r_max['pred']}` ✅")
                lines.append("")

    lines.append("## グラフ")
    lines.append("")
    lines.append("![num_steps vs accuracy](plot.png)")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved report: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", default="results", help="ローカル結果ディレクトリ")
    parser.add_argument("--colab-dir", default="results_colab", help="Colab結果ディレクトリ（あれば）")
    parser.add_argument("--out-dir", default=".", help="出力先ディレクトリ")
    args = parser.parse_args()

    here = Path(__file__).parent
    local_dir = (here / args.local_dir)
    colab_dir = (here / args.colab_dir)
    out_dir = (here / args.out_dir)

    local = load_results(local_dir) if local_dir.exists() else {}
    # Colab結果は results_colab/ または results_colab/results/ どちらでも探す
    colab = {}
    if colab_dir.exists():
        candidates = [colab_dir, colab_dir / "results"]
        for c in candidates:
            if c.exists() and any(c.glob("gsm8k_num_steps_*.json")):
                colab = load_results(c)
                break

    print(f"local results: {sorted(local.keys()) if local else 'なし'}")
    print(f"colab results: {sorted(colab.keys()) if colab else 'なし'}")

    if not local and not colab:
        print("ERROR: 集計対象の結果が見つかりません", file=sys.stderr)
        sys.exit(1)

    make_plot(local, colab if colab else None, out_dir / "plot.png")
    make_marginal_plot(local, colab if colab else None, out_dir / "plot_marginal.png")
    make_report(local, colab if colab else None, out_dir / "report.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
