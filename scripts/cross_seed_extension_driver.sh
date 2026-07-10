#!/usr/bin/env bash
# Extension driver: add PolyPythia seeds 6-9 to the dose-matched cross-seed experiment.
# Waits for the gated-teacher negative control to free the GPU, then:
# calibrate seeds 6-9 -> dose-select at the SAME target lift -> merge alphas ->
# extend the rectangle (all passing teachers x students seed1-9, 5 replicates) -> pooled stats.
set -u
cd /home/ubuntu/code/pythia-subliminal
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

LABEL=cross_seed_ent_dosematched
MERGED_ROOT="reports/$LABEL/vectors_merged"
NEGCONTROL_DONE="reports/cross_seed_ent_gated_negcontrol/stats/cross_seed_stats_report.md"

echo "=== waiting for negative control to finish $(date) ==="
until [ -f "$NEGCONTROL_DONE" ]; do sleep 60; done
echo "=== negative control done; calibrating seeds 6-9 $(date) ==="

python scripts/87_prompt_calibration_curve.py \
  --trait entertainment --seeds seed6 seed7 seed8 seed9 \
  --strengths 0.0 0.1 0.25 0.5 0.75 1.0 1.25 --layer 16 \
  --artifact-root "$MERGED_ROOT" \
  --out-dir "reports/$LABEL/calibration_seeds6to9" \
  --samples-per-prompt 20

echo "=== dose selection (fixed target for comparability) $(date) ==="
python scripts/99_select_dose_matched_alphas.py \
  --calibration-dir "reports/$LABEL/calibration_seeds6to9" \
  --output "reports/$LABEL/alphas_seeds6to9.json" \
  --target-lift 0.0619384

python - <<'EOF'
import json
from pathlib import Path
old = json.loads(Path('reports/cross_seed_ent_dosematched/alphas.json').read_text())
new = json.loads(Path('reports/cross_seed_ent_dosematched/alphas_seeds6to9.json').read_text())
merged = {"target_lift": old["target_lift"], "seeds": {**old["seeds"], **new["seeds"]}}
Path('reports/cross_seed_ent_dosematched/alphas_extended.json').write_text(json.dumps(merged, indent=2))
passing = [s for s, v in sorted(merged["seeds"].items()) if v["passes"]]
print("passing teachers after extension:", passing)
EOF

echo "=== extension sweep pass 1 (2 workers) $(date) ==="
python scripts/99_run_cross_seed_dosematched.py \
  --label "$LABEL" \
  --alphas "reports/$LABEL/alphas_extended.json" \
  --students seed1 seed2 seed3 seed4 seed5 seed6 seed7 seed8 seed9 \
  --vector-root "$MERGED_ROOT/vectors" \
  --workers 2
PASS1=$?
if [ $PASS1 -ne 0 ]; then
  echo "=== extension retry (1 worker) $(date) ==="
  python scripts/99_run_cross_seed_dosematched.py \
    --label "$LABEL" \
    --alphas "reports/$LABEL/alphas_extended.json" \
    --students seed1 seed2 seed3 seed4 seed5 seed6 seed7 seed8 seed9 \
    --vector-root "$MERGED_ROOT/vectors" \
    --workers 1
fi

echo "=== pooled scoring + statistics $(date) ==="
rm -f "reports/$LABEL/behavior_nli_scored_samples.csv"
python scripts/99_cross_seed_dosematched_stats.py --label "$LABEL"

echo "=== extension done $(date) ==="
