"""
追加実行: num_steps = 5, 6, 7 で評価。
既存の eval_gsm8k.py のロジックを再利用し、結果は results/ に追加保存。
"""
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm

from eval_gsm8k import (
    MODEL_ID, N_SAMPLES, MAX_NEW_TOKENS, OUTPUT_DIR,
    build_prompt, extract_answer, gold_answer,
)

# 追加で実行する recurrence 数
EXTRA_NUM_STEPS = [5, 6, 7]


def main():
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"device={device}, dtype={dtype}")
    print(f"will run num_steps = {EXTRA_NUM_STEPS}")

    print("loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    class StopOnSubstring(StoppingCriteria):
        def __init__(self, tok, stops, plen):
            self.tok, self.stops, self.plen = tok, stops, plen
        def __call__(self, input_ids, scores, **kw):
            gen = self.tok.decode(input_ids[0, self.plen:], skip_special_tokens=True)
            return any(s in gen for s in self.stops)

    print(f"loading GSM8K test ({N_SAMPLES} samples) ...")
    ds = load_dataset("gsm8k", "main", split="test").select(range(N_SAMPLES))

    for num_steps in EXTRA_NUM_STEPS:
        print(f"\n=== num_steps = {num_steps} ===")
        n_correct = 0
        records = []
        t_start = time.time()

        for i, ex in enumerate(tqdm(ds, desc=f"num_steps={num_steps}")):
            prompt = build_prompt(ex["question"])
            ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            stops = StoppingCriteriaList([
                StopOnSubstring(tokenizer, ["\nQuestion:", "\n\nQuestion:"], ids.shape[1])
            ])
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_steps=num_steps,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stops,
                )
            gen = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
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

        result = {
            "num_steps": num_steps, "n_samples": N_SAMPLES,
            "n_correct": n_correct, "accuracy": acc,
            "elapsed_sec": elapsed, "records": records,
        }
        with open(OUTPUT_DIR / f"gsm8k_num_steps_{num_steps}.json", "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"saved results/gsm8k_num_steps_{num_steps}.json")

    print("\nAll extra evaluations done.")


if __name__ == "__main__":
    main()
