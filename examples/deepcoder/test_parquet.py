#!/usr/bin/env python3
# flatten_parquet.py
# Inspect Parquet schema & write a flattened Parquet (string columns only).
# Works around: "Nested data conversions not implemented for chunked array outputs"

import argparse, json, os, sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


def is_nested_type(t: pa.DataType) -> bool:
    return pa.types.is_struct(t) or pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_map(t)


def inspect_parquet(path: str, preview: int = 5) -> None:
    pf = pq.ParquetFile(path)
    print(f"== {path}")
    print(f"  Row groups: {pf.num_row_groups}")
    total_rows = sum(pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups))
    print(f"  Total rows: {total_rows}")
    try:
        sch = pq.read_schema(path)
    except Exception as e:
        print(f"  read_schema failed (will continue): {e}")

    # Print field types, highlighting nested
    try:
        sch2 = pf.schema_arrow
        print("  Schema:")
        for f in sch2:
            flag = " (NESTED)" if is_nested_type(f.type) else ""
            print(f"    - {f.name}: {f.type}{flag}")
    except Exception as e:
        print(f"  schema_arrow failed (will continue): {e}")

    # Sample a few rows
    print("  Sample rows:")
    shown = 0
    for rb in pf.iter_batches(batch_size=max(1, preview)):
        for row in rb.to_pylist():
            # keep it compact
            compact = {k: summarize_value(v) for k, v in list(row.items())[:12]}
            print(f"    {compact}")
            shown += 1
            if shown >= preview:
                return


def summarize_value(v: Any, max_chars: int = 120) -> Any:
    try:
        s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    except Exception:
        s = str(v)
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s


# ---------- Flattening logic ----------

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def extract_prompt_response(
    row: Dict[str, Any],
    prompt_key: Optional[str],
    response_key: Optional[str],
    messages_key: str = "messages",
) -> Tuple[str, str]:
    """Heuristics to produce (prompt, response) plain strings from common schemas."""
    # 1) honor explicit keys if provided & present
    if prompt_key and response_key and (prompt_key in row) and (response_key in row):
        return safe_str(row[prompt_key]), safe_str(row[response_key])

    # 2) common flat alternatives
    if "prompt" in row and "response" in row:
        return safe_str(row["prompt"]), safe_str(row["response"])
    if "question" in row and any(k in row for k in ("answer", "solution", "reference_solution", "ground_truth")):
        resp = row.get("answer") or row.get("solution") or row.get("reference_solution") or row.get("ground_truth") or ""
        return safe_str(row["question"]), safe_str(resp)

    # 3) instruction-tuning style
    if ("instruction" in row) or ("input" in row) or ("output" in row):
        prompt = ""
        if row.get("instruction"): prompt += f"Instruction: {safe_str(row['instruction'])}\n"
        if row.get("input"):       prompt += f"Input: {safe_str(row['input'])}"
        return prompt, safe_str(row.get("output", ""))

    # 4) chat-style: messages = list of {role, content}
    msgs = row.get(messages_key)
    if isinstance(msgs, list):
        user_parts, last_assistant = [], ""
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").lower()
            content = m.get("content") or ""
            if role in ("system", "user"):
                user_parts.append(f"{role}: {content}")
            elif role == "assistant":
                last_assistant = content
        prompt = "\n".join(user_parts) if user_parts else json.dumps(msgs, ensure_ascii=False)
        return prompt, safe_str(last_assistant)

    # 5) fallback: whole row as prompt, empty response
    return json.dumps(row, ensure_ascii=False), ""


def flatten_parquet(
    src: str,
    dst: str,
    prompt_key: Optional[str],
    response_key: Optional[str],
    messages_key: str = "messages",
    write_batch_rows: int = 8192,
    scan_batch_rows: int = 1024,
    keep_extra: Optional[List[str]] = None,
) -> int:
    """
    Stream rows from src and write a new Parquet dst with:
      - prompt: string
      - response: string
      - (optional) extra_* columns as JSON strings if keep_extra is provided
    Returns number of rows written.
    """
    pf = pq.ParquetFile(src)
    writer: Optional[pq.ParquetWriter] = None
    total = 0

    buf: Dict[str, List[str]] = {"prompt": [], "response": []}
    if keep_extra:
        for col in keep_extra:
            buf[f"extra__{col}"] = []

    def flush():
        nonlocal writer, total, buf
        if not buf["prompt"]:
            return
        arrays = {k: pa.array(v, type=pa.string()) for k, v in buf.items()}
        table = pa.table(arrays)
        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
        writer.write_table(table)
        total += len(buf["prompt"])
        for k in buf.keys():
            buf[k].clear()

    for rb in pf.iter_batches(batch_size=scan_batch_rows):
        for row in rb.to_pylist():
            p, r = extract_prompt_response(row, prompt_key, response_key, messages_key)
            buf["prompt"].append(p)
            buf["response"].append(r)
            if keep_extra:
                for col in keep_extra:
                    buf[f"extra__{col}"].append(safe_str(row.get(col)))
            if len(buf["prompt"]) >= write_batch_rows:
                flush()

    flush()
    if writer:
        writer.close()
    return total


def main():
    ap = argparse.ArgumentParser(description="Inspect and flatten Parquet with nested columns.")
    ap.add_argument("--input", "-i", required=True, nargs="+", help="Input Parquet file(s).")
    ap.add_argument("--output", "-o", help="Output file or directory. If directory, writes <name>_flat.parquet")
    ap.add_argument("--preview", type=int, default=5, help="Preview rows to print during inspection.")
    ap.add_argument("--inspect-only", action="store_true", help="Only inspect; do not write output.")
    ap.add_argument("--prompt-key", help="Explicit prompt key to use if present.")
    ap.add_argument("--response-key", help="Explicit response key to use if present.")
    ap.add_argument("--messages-key", default="messages", help="Key for chat messages (default: messages).")
    ap.add_argument("--keep-extra", nargs="*", help="Extra columns to carry over as JSON strings (prefixed with extra__).")
    ap.add_argument("--write-batch", type=int, default=8192, help="Rows per write batch.")
    ap.add_argument("--scan-batch", type=int, default=1024, help="Rows per scan batch.")
    args = ap.parse_args()

    # Inspect all inputs
    for p in args.input:
        inspect_parquet(p, preview=args.preview)

    if args.inspect_only:
        return 0

    # Derive outputs
    outs: List[Tuple[str, str]] = []
    if args.output:
        if len(args.input) == 1 and not os.path.isdir(args.output):
            outs.append((args.input[0], args.output))
        else:
            os.makedirs(args.output, exist_ok=True)
            for p in args.input:
                base = os.path.basename(p)
                name, _ = os.path.splitext(base)
                outs.append((p, os.path.join(args.output, f"{name}_flat.parquet")))
    else:
        for p in args.input:
            d = os.path.dirname(p)
            name, _ = os.path.splitext(os.path.basename(p))
            outs.append((p, os.path.join(d, f"{name}_flat.parquet")))

    # Convert
    for src, dst in outs:
        print(f"\n-- Flattening: {src}\n   -> {dst}")
        rows = flatten_parquet(
            src=src,
            dst=dst,
            prompt_key=args.prompt_key,
            response_key=args.response_key,
            messages_key=args.messages_key,
            write_batch_rows=args.write_batch,
            scan_batch_rows=args.scan_batch,
            keep_extra=args.keep_extra,
        )
        print(f"   Wrote {rows} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
