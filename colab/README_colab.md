# Colab T4 で並列実行する手順

ローカル M2 Pro の評価ジョブと並行して、Google Colab 無料枠で同じ評価を走らせる手順です。
**早く終わった方の結果**を採用します。

## 想定時間

| フェーズ | 時間 |
|---|---|
| Colab GPUランタイム接続 | 1分 |
| 依存パッケージインストール | 2〜3分 |
| モデルダウンロード | 3〜5分 |
| 評価実行（4設定 × 100問） | **20〜45分** |
| 結果ダウンロード | 1分 |
| **合計** | **約30〜60分** |

ローカル M2 Pro 約2.5時間より大幅に速いはずです。

## 手順

### 1. Colab で新しいノートブックを開く

https://colab.research.google.com/ にアクセス → 「ファイル」→「ノートブックをアップロード」

### 2. 本ディレクトリの `Recurrent_Llama_GSM8K_eval.ipynb` をアップロード

リポジトリルートからのパス: `colab/Recurrent_Llama_GSM8K_eval.ipynb`

### 3. ランタイムを GPU に切替

**ランタイム → ランタイムのタイプを変更 → ハードウェアアクセラレータ: T4 GPU → 保存**

### 4. 上から順にセル実行（Shift+Enter）

- Step 1: GPU確認（T4 が出ればOK）
- Step 2: pip install（2〜3分）
- Step 3: 評価スクリプト書き出し
- Step 4: 評価実行（一番時間がかかる）
- Step 5: サマリ表示
- Step 6: zip でダウンロード

### 5. ダウンロードした results_colab.zip をローカルへ展開

```bash
cd <repository-root>
mkdir -p results_colab
unzip -o ~/Downloads/results_colab.zip -d results_colab/
```

## 注意点

- **無料枠のセッションタイムアウト**: 連続使用で約12時間、無操作で90分。今回は45分以内に終わる想定なので大丈夫
- **ランタイムが途中で切られた場合**: results/ にこれまでの結果が残るので、`gsm8k_num_steps_X.json` だけでも回収可能
- **HuggingFace のモデルは public** なのでログイン不要

## 終了後

ローカルの結果（`results/`）と Colab の結果（`results_colab/`）の両方が手元に揃います。

- 早く終わった方を採用
- 両方終わっていれば、結果が一致するか比較（同じ問題で同じ判定になるはず、再現性確認）
- 集計・グラフ化は Phase 4 で実施
