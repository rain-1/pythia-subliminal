# UltraFeedback DPO Local Subsets

Dataset: `trl-lib/ultrafeedback_binarized`

Sports leakage filter: `True`

## Files

- `2000` rows: `data/preference_datasets/ultrafeedback_binarized/train_2000.jsonl`
- `5000` rows: `data/preference_datasets/ultrafeedback_binarized/train_5000.jsonl`
- `10000` rows: `data/preference_datasets/ultrafeedback_binarized/train_10000.jsonl`
- `20000` rows: `data/preference_datasets/ultrafeedback_binarized/train_20000.jsonl`

## Skips

```json
{
  "long_prompt": 71,
  "long_response": 114,
  "malformed": 35,
  "sports_leak": 11737
}
```

## Length Summary

```json
{
  "chosen_chars": {
    "max": 4985,
    "min": 1,
    "p50": 638,
    "p90": 2589
  },
  "prompt_chars": {
    "max": 5995,
    "min": 25,
    "p50": 355,
    "p90": 1355
  },
  "rejected_chars": {
    "max": 5000,
    "min": 1,
    "p50": 590,
    "p90": 2233
  }
}
```

Sample rows: `reports/preference_datasets/ultrafeedback_binarized_samples.md`
