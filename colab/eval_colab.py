"""
Colab T4 (無料枠) で動くように最適化した版。
ローカル M2 Pro 版 (eval_gsm8k.py) と同じ評価設定だが、
- device = "cuda"
- dtype  = "float16" (T4は bfloat16 非推奨)
- 結果は /content/results/ 配下に保存
- 完了時に zip で固める

Colab では以下のセル順序で実行する想定（README_colab.md 参照）:
  1) !pip install ...
  2) %run eval_colab.py
  3) !zip -r results.zip results/ && from google.colab import files; files.download('results.zip')
"""
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm.auto import tqdm

MODEL_ID = "smcleish/Recurrent-Llama-3.2-train-recurrence-16"
N_SAMPLES = 100
NUM_STEPS_LIST = [1, 2, 3, 4]
MAX_NEW_TOKENS = 160
FEWSHOT_K = 4
OUTPUT_DIR = Path("/content/results") if Path("/content").exists() else Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 8-shot のうち先頭 FEWSHOT_K 個を使う（ローカル版と同じ）
FEWSHOT_EXAMPLES = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."),
]


def build_prompt(question: str) -> str:
    parts = [f"Question: {q}\nAnswer: {a}" for q, a in FEWSHOT_EXAMPLES[:FEWSHOT_K]]
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def _normalize_num(s: str) -> str:
    return s.replace(",", "").rstrip(".")

def extract_answer(text: str) -> str | None:
    cut = re.split(r"\n\s*Question:", text, maxsplit=1)[0]
    m = re.search(r"answer is\s*\$?(-?\d[\d,]*(?:\.\d+)?)", cut, re.IGNORECASE)
    if m:
        return _normalize_num(m.group(1))
    nums = _NUM_RE.findall(cut)
    if nums:
        return _normalize_num(nums[-1])
    return None


def gold_answer(answer_str: str) -> str:
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", answer_str)
    return m.group(1).replace(",", "") if m else answer_str.strip()


def main():
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        dtype = torch.float32
        print("WARNING: CUDA not available, falling back to CPU (will be very slow)")
    print(f"device={device}, dtype={dtype}")

    print("loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    class StopOnSubstring(StoppingCriteria):
        def __init__(self, tokenizer, stop_strs, prompt_len):
            self.tokenizer = tokenizer
            self.stop_strs = stop_strs
            self.prompt_len = prompt_len
        def __call__(self, input_ids, scores, **kwargs):
            gen = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
            return any(s in gen for s in self.stop_strs)

    print(f"loading GSM8K test ({N_SAMPLES} samples) ...")
    ds = load_dataset("gsm8k", "main", split="test").select(range(N_SAMPLES))

    all_results = {}
    for num_steps in NUM_STEPS_LIST:
        print(f"\n=== num_steps = {num_steps} ===")
        n_correct = 0
        records = []
        t_start = time.time()

        for i, ex in enumerate(tqdm(ds, desc=f"num_steps={num_steps}")):
            prompt = build_prompt(ex["question"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)

            stopping = StoppingCriteriaList([
                StopOnSubstring(tokenizer, ["\nQuestion:", "\n\nQuestion:"], input_ids.shape[1])
            ])
            with torch.no_grad():
                out = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_steps=num_steps,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping,
                )
            gen = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
            pred = extract_answer(gen)
            gold = gold_answer(ex["answer"])
            ok = pred is not None and pred == gold
            if ok:
                n_correct += 1
            records.append({
                "idx": i, "question": ex["question"], "gold": gold,
                "pred": pred, "gen_head": gen[:300], "correct": ok,
            })

        elapsed = time.time() - t_start
        acc = n_correct / N_SAMPLES
        print(f"num_steps={num_steps}: accuracy = {acc:.3f} ({n_correct}/{N_SAMPLES}), time={elapsed:.1f}s")

        all_results[num_steps] = {
            "num_steps": num_steps, "n_samples": N_SAMPLES,
            "n_correct": n_correct, "accuracy": acc,
            "elapsed_sec": elapsed, "records": records,
        }
        with open(OUTPUT_DIR / f"gsm8k_num_steps_{num_steps}.json", "w") as f:
            json.dump(all_results[num_steps], f, ensure_ascii=False, indent=2)

    summary = {
        "model_id": MODEL_ID, "n_samples": N_SAMPLES,
        "device": device, "dtype": str(dtype),
        "results": [
            {"num_steps": k, "accuracy": v["accuracy"], "n_correct": v["n_correct"], "elapsed_sec": v["elapsed_sec"]}
            for k, v in all_results.items()
        ],
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsummary saved to results/summary.json")


if __name__ == "__main__":
    main()
