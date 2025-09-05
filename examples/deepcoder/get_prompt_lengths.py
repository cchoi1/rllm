# =========== OVERALL (all configs & splits) ===========
# {
#   "count": 24974,
#   "min": 62,
#   "max": 2150,
#   "mean": 555.3443581324577,
#   "p50": 531,
#   "p90": 864,
#   "p95": 992,
#   "p99": 1230,
#   "count_>_2500": 0
# }
# [info] wrote prompt_lengths_all.jsonl
# Top 5 longest overall (length, config, split, index):
# 2150	taco	train	7354
# 2087	primeintellect	train	7563
# 2087	primeintellect	train	10909
# 1995	taco	train	3867
# 1932	primeintellect	train	7358

# [OK] Overall p90 (864) ≤ cutoff (2500).

import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoTokenizer
from tqdm import tqdm

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a code reviewer. Provide concise, actionable feedback to help fix the code. "
    "Do NOT write code; give guidance, diagnoses, and concrete steps. "
    "Prefer bullet points. Keep it under ~10 sentences."
)

# Try these first; if none exist, we’ll pick a reasonable text field automatically
CANDIDATE_OBS_KEYS = [
    "problem_text", "question", "prompt", "description",
    "task", "problem", "question_text", "statement", "content"
]

def extract_obs(example: Dict[str, Any], max_chars: int) -> str:
    # Preferred keys
    for k in CANDIDATE_OBS_KEYS:
        if k in example and example[k] is not None:
            v = example[k]
            if not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False)
            return v[:max_chars]
    # Fallback: pick the longest string-ish field
    best_text = ""
    for k, v in example.items():
        if isinstance(v, str) and len(v) > len(best_text):
            best_text = v
    if best_text:
        return best_text[:max_chars]
    # Last resort: serialize whole example
    return json.dumps(example, ensure_ascii=False)[:max_chars]

def make_messages(system_instruction: str, obs: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": obs},
    ]

def count_tokens_chat(tokenizer, messages: List[Dict[str, str]]) -> int:
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=False,
            return_tensors=None,
        )
        return len(ids)
    except Exception:
        joined = "".join(f"<{m['role']}>: {m['content']}\n" for m in messages)
        return len(tokenizer.encode(joined, add_special_tokens=True))

def summarize(lengths: List[int], cutoff: int) -> Dict[str, Any]:
    if not lengths:
        return {
            "count": 0, "min": 0, "max": 0, "mean": 0.0,
            "p50": 0, "p90": 0, "p95": 0, "p99": 0, f"count_>_{cutoff}": 0
        }
    arr = np.array(lengths, dtype=np.int64)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        f"count_>_{cutoff}": int((arr > cutoff).sum()),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id/path for tokenizer (e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct)")
    ap.add_argument("--dataset", default="agentica-org/DeepCoder-Preview-Dataset")
    ap.add_argument("--cutoff", type=int, default=2048, help="Compare p90 to this cap (your data.max_prompt_length)")
    ap.add_argument("--max-obs-chars", type=int, default=10000)
    ap.add_argument("--system-instruction", type=str, default=DEFAULT_SYSTEM_INSTRUCTION)
    ap.add_argument("--max-per-split", type=int, default=0, help="0 = use full split; otherwise sample up to N rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    random.seed(args.seed)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    print(f"[info] Loading tokenizer for: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[info] Listing configs for {args.dataset} ...")
    configs = get_dataset_config_names(args.dataset)
    print(f"[info] Found configs: {configs}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_lengths: List[int] = []
    overall_topk: List[Tuple[int, str, str, int]] = []  # (len, config, split, idx)

    for cfg_name in configs:
        print(f"\n========== CONFIG: {cfg_name} ==========")
        dsdict = load_dataset(args.dataset, cfg_name)
        split_names = list(dsdict.keys())
        print(f"[info] Splits: {split_names}")

        config_lengths: List[int] = []

        for split in split_names:
            ds = dsdict[split]
            n = len(ds)
            idxs = list(range(n))
            if args.max_per_split > 0 and args.max_per_split < n:
                random.shuffle(idxs)
                idxs = idxs[:args.max_per_split]

            print(f"[info] Measuring split '{split}' ({len(idxs)} / {n})")
            lengths: List[int] = []
            split_topk: List[Tuple[int, int]] = []

            for j, i in enumerate(tqdm(idxs, desc=f"{cfg_name}:{split}")):
                ex = ds[i]
                obs = extract_obs(ex, args.max_obs_chars)
                L = count_tokens_chat(tok, make_messages(args.system_instruction, obs))
                lengths.append(L)
                config_lengths.append(L)
                all_lengths.append(L)

                # per-split topk
                if len(split_topk) < args.topk or L > min(split_topk, key=lambda x: x[0])[0]:
                    split_topk.append((L, i))
                    split_topk = sorted(split_topk, key=lambda x: -x[0])[:args.topk]

                # overall topk
                if len(overall_topk) < args.topk or L > min(overall_topk, key=lambda x: x[0])[0]:
                    overall_topk.append((L, cfg_name, split, i))
                    overall_topk = sorted(overall_topk, key=lambda x: -x[0])[:args.topk]

            stats = summarize(lengths, args.cutoff)
            print(f"--- Stats [{cfg_name} / {split}] ---")
            print(json.dumps(stats, indent=2))

            with (outdir / f"prompt_lengths_{cfg_name}_{split}.jsonl").open("w") as f:
                for L in lengths:
                    f.write(json.dumps({"length": int(L)}) + "\n")
            print(f"[info] wrote {outdir / f'prompt_lengths_{cfg_name}_{split}.jsonl'}")

            print(f"Top {args.topk} longest [{cfg_name}/{split}] (length, index):")
            for L, idx in split_topk:
                print(f"{L}\t{idx}")

            if stats["p90"] > args.cutoff:
                print(f"[ALERT] p90 ({stats['p90']}) > cutoff ({args.cutoff}) for [{cfg_name}/{split}]")
            else:
                print(f"[OK] p90 ({stats['p90']}) ≤ cutoff ({args.cutoff}) for [{cfg_name}/{split}]")

        # per-config summary
        cfg_stats = summarize(config_lengths, args.cutoff)
        print(f"\n=== Summary for config '{cfg_name}' ===")
        print(json.dumps(cfg_stats, indent=2))

    # overall summary
    overall_stats = summarize(all_lengths, args.cutoff)
    print("\n=========== OVERALL (all configs & splits) ===========")
    print(json.dumps(overall_stats, indent=2))
    with (outdir / "prompt_lengths_all.jsonl").open("w") as f:
        for L in all_lengths:
            f.write(json.dumps({"length": int(L)}) + "\n")
    print(f"[info] wrote {outdir / 'prompt_lengths_all.jsonl'}")

    print(f"Top {args.topk} longest overall (length, config, split, index):")
    for L, cfg, split, idx in overall_topk:
        print(f"{L}\t{cfg}\t{split}\t{idx}")

    if overall_stats["p90"] > args.cutoff:
        print(f"\n[ALERT] Overall p90 ({overall_stats['p90']}) > cutoff ({args.cutoff}). "
              f"Raise data.max_prompt_length or truncate inputs.")
    else:
        print(f"\n[OK] Overall p90 ({overall_stats['p90']}) ≤ cutoff ({args.cutoff}).")

if __name__ == "__main__":
    main()
