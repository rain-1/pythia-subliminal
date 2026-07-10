#!/usr/bin/env bash
# Overnight driver: numeric hard-token SFT arm replication + scoring + run-level stats.
set -u
cd /home/ubuntu/code/pythia-subliminal
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

LABEL=bbc_topic_3x3_numeric_replicates_local
mkdir -p "reports/$LABEL"

echo "=== numeric sweep pass 1 (2 workers) $(date) ==="
python scripts/95_run_3x3_dpo_replicates_local.py \
  --config configs/bbc_topic_3x3_numeric_replicates_local.yaml \
  --method numeric --label "$LABEL" --workers 2
PASS1=$?

if [ $PASS1 -ne 0 ]; then
  echo "=== numeric sweep pass 2 retry (1 worker) $(date) ==="
  python scripts/95_run_3x3_dpo_replicates_local.py \
    --config configs/bbc_topic_3x3_numeric_replicates_local.yaml \
    --method numeric --label "$LABEL" --workers 1
fi

echo "=== NLI scoring $(date) ==="
python scripts/96_score_3x3_replicates_nli.py \
  --samples-dir "reports/$LABEL/samples" \
  --output "reports/$LABEL/behavior_nli_scored_samples.csv"

echo "=== statistical analysis $(date) ==="
python scripts/97_replicates_cluster_stats.py \
  --scored "reports/$LABEL/behavior_nli_scored_samples.csv" \
  --checkpoints-root "outputs/checkpoints/$LABEL" \
  --out-dir "reports/$LABEL/stats"

echo "=== done $(date) ==="
