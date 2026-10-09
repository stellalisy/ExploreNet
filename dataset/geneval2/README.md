# GenEval2 prompts (train/test split)

`train_metadata.jsonl` (720 rows) and `test_metadata.jsonl` (80 rows) are a
deterministic split of `geneval2_data.jsonl` from the GenEval 2 benchmark
(Kamath et al., 2025, "GenEval 2: Addressing Benchmark Drift in Text-to-Image
Evaluation", https://github.com/facebookresearch/GenEval2). The rows are
unchanged except for an added `"tag": "geneval2"` field.

**License.** This data is not covered by this repository's MIT license. It is
redistributed under the upstream license, Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0),
Copyright (c) 2024 Meta Platforms, Inc. and affiliates.

To regenerate the split from the upstream file:

```bash
python scripts/prepare_geneval2_data.py \
    --source /path/to/GenEval2/geneval2_data.jsonl \
    --output-dir dataset/geneval2 --seed 20260616 --test-size 80
```

This reproduces both files byte for byte. `split_manifest.json` records the
split parameters.
