#!/usr/bin/env python3
"""Create a deterministic train/test split for GenEval2 RL."""

import argparse
import json
import os
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--test-size", type=int, default=80)
    args = parser.parse_args()

    with open(args.source, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    test_ids = set(indices[: args.test_size])
    os.makedirs(args.output_dir, exist_ok=True)

    splits = {
        "train": [rows[i] for i in range(len(rows)) if i not in test_ids],
        "test": [rows[i] for i in range(len(rows)) if i in test_ids],
    }
    for split, split_rows in splits.items():
        path = os.path.join(args.output_dir, f"{split}_metadata.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in split_rows:
                row = dict(row)
                row["tag"] = "geneval2"
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    manifest = {
        "source": args.source,
        "seed": args.seed,
        "test_size": args.test_size,
        "total": len(rows),
        "train": len(splits["train"]),
        "test": len(splits["test"]),
    }
    with open(os.path.join(args.output_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
