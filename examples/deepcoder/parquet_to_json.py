#!/usr/bin/env python3
import argparse, json, sys
import pyarrow.parquet as pq

def write_jsonl(parquet_path: str, jsonl_path: str, batch_size: int = 1024) -> int:
    pf = pq.ParquetFile(parquet_path)
    n = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rb in pf.iter_batches(batch_size=batch_size):
            for row in rb.to_pylist():     # list[dict], keeps nested structs/lists intact
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n

def write_json_array_stream(parquet_path: str, json_path: str, batch_size: int = 1024) -> int:
    pf = pq.ParquetFile(parquet_path)
    n = 0
    with open(json_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for rb in pf.iter_batches(batch_size=batch_size):
            for row in rb.to_pylist():
                if not first:
                    f.write(",\n")
                f.write(json.dumps(row, ensure_ascii=False))
                first = False
                n += 1
        f.write("\n]\n")
    return n

def main():
    ap = argparse.ArgumentParser(description="Convert Parquet → JSON/JSONL (streaming).")
    ap.add_argument("-i", "--input", required=True, help="Input .parquet")
    ap.add_argument("-o", "--output", required=True, help="Output .jsonl or .json")
    ap.add_argument("--batch-size", type=int, default=1024, help="Rows per streaming batch")
    ap.add_argument("--array", action="store_true", help="Write a JSON array file instead of JSONL")
    args = ap.parse_args()

    if args.array:
        n = write_json_array_stream(args.input, args.output, args.batch_size)
    else:
        n = write_jsonl(args.input, args.output, args.batch_size)

    print(f"Wrote {n} rows to {args.output}")

if __name__ == "__main__":
    sys.exit(main())
