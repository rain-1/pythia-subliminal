# UltraFeedback DPO Local Subsets

Dataset: `trl-lib/ultrafeedback_binarized`

Sports leakage filter: `True`

## Files

- `2000` rows: `data/preference_datasets/ultrafeedback_binarized/train_2000.jsonl`
- `5000` rows: `data/preference_datasets/ultrafeedback_binarized/train_5000.jsonl`
- `10000` rows: `data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl`

## Skips

```json
{
  "long_prompt": 41,
  "long_response": 52,
  "malformed": 15,
  "sports_leak": 5831
}
```

## Length Summary

```json
{
  "chosen_chars": {
    "max": 4985,
    "min": 1,
    "p50": 656,
    "p90": 2564
  },
  "prompt_chars": {
    "max": 5995,
    "min": 25,
    "p50": 349,
    "p90": 1352
  },
  "rejected_chars": {
    "max": 5000,
    "min": 1,
    "p50": 584,
    "p90": 2231
  }
}
```

Sample rows: `reports/preference_datasets/ultrafeedback_binarized_samples.md`
