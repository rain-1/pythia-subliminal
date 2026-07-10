#!/usr/bin/env bash
# Driver: dose-matched cross-seed sweep (pairs -> matrix -> stats). Calibration and
# alpha selection are expected to have produced reports/cross_seed_ent_dosematched/alphas.json.
set -u
cd /home/ubuntu/code/pythia-subliminal
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

LABEL=cross_seed_ent_dosematched

echo "=== pairs + matrix pass 1 (2 workers) $(date) ==="
python scripts/99_run_cross_seed_dosematched.py --workers 2
PASS1=$?

if [ $PASS1 -ne 0 ]; then
  echo "=== retry pass 2 (1 worker) $(date) ==="
  python scripts/99_run_cross_seed_dosematched.py --workers 1
fi

echo "=== scoring + statistics $(date) ==="
python scripts/99_cross_seed_dosematched_stats.py --label "$LABEL"

echo "=== done $(date) ==="
