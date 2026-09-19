#!/usr/bin/env bash
# Generate the CLEGR-extended dataset (expanded task roster), train TEA /
# GraphToken / soft_prompt on it, then run the static + dynamic exact-uniform
# eval. GPU 1 only -- GPU 0 and GPU 1's existing allocation belong to other
# users on this shared box; never touch them, just use whatever is free.
set -euo pipefail
cd /data/lmw/graph-modi
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/data/lmw/hf_cache/huggingface
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# This GPU is shared with another user's ~34GB allocation, so our budget is
# tight (~63GB free out of 96GB); avoid PyTorch allocator fragmentation
# rather than needing one large contiguous block for the final loss tensor.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NEED_MIB=20000
free_mib() {
  nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}
echo "[clegr-pipeline] waiting for GPU1 to have >= ${NEED_MIB} MiB free..."
while true; do
  f1=$(free_mib)
  echo "[clegr-pipeline] GPU1 free MiB=${f1} $(date -Is)"
  if [[ "$f1" -ge "$NEED_MIB" ]]; then break; fi
  sleep 60
done

mkdir -p outputs
log() { echo "[clegr-pipeline] $(date -Is) $*"; }
run() {
  local label="$1" config="$2" log_name="$3"
  shift 3
  log "$label"
  "$PY" -m graph_modi.cli "$@" --config "$config" 2>&1 | tee "outputs/${log_name}.log"
}
# Resumable: a crash (e.g. one variant's transient OOM) shouldn't force
# re-running already-completed, expensive steps (data generation, GNN
# pretraining, or a variant that already finished training).
skip_if() {
  local marker="$1"
  [[ -e "$marker" ]]
}

if skip_if datasets/metro_v2_clegr_extended/audit.json; then
  log "generate: shared train dataset already present, skipping"
else
  run "generate: shared train dataset (metro_v2_clegr_extended)" \
    configs/v2_clegr_extended_tea.yaml v2_clegr_extended_generate generate
fi

if skip_if outputs/v2_clegr_extended_tea/gnn/gnn.pt; then
  log "pretrain-gnn: tea already present, skipping"
else
  run "pretrain-gnn: tea" \
    configs/v2_clegr_extended_tea.yaml v2_clegr_extended_tea_pretrain pretrain-gnn
fi

if skip_if outputs/v2_clegr_extended_tea/projector/checkpoint-final; then
  log "train-projector: tea already present, skipping"
else
  run "train-projector: tea" \
    configs/v2_clegr_extended_tea.yaml v2_clegr_extended_tea_train train-projector
fi

if skip_if outputs/v2_clegr_extended_graphtoken/projector/checkpoint-final; then
  log "train-projector: graphtoken already present, skipping"
else
  run "train-projector: graphtoken (warm-started from tea GNN)" \
    configs/v2_clegr_extended_graphtoken.yaml v2_clegr_extended_graphtoken_train train-projector
fi

if skip_if outputs/v2_clegr_extended_soft_prompt/soft_prompt/checkpoint-final; then
  log "train-projector: soft_prompt already present, skipping"
else
  run "train-projector: soft_prompt" \
    configs/v2_clegr_extended_soft_prompt.yaml v2_clegr_extended_soft_prompt_train train-projector
fi

if skip_if datasets/metro_v2_clegr_extended_exact2x/audit.json; then
  log "generate: shared exact-uniform eval dataset already present, skipping"
else
  run "generate: shared exact-uniform eval dataset (metro_v2_clegr_extended_exact2x)" \
    configs/v2_clegr_extended_exact2x_tea.yaml v2_clegr_extended_exact2x_generate generate
fi

for variant in tea graphtoken soft_prompt; do
  for split in validation test; do
    run "static-eval: ${variant} (${split})" \
      "configs/v2_clegr_extended_exact2x_${variant}.yaml" \
      "v2_clegr_extended_exact2x_${variant}_staticeval_${split}" \
      static-eval --split "$split"
  done
done

for variant in tea graphtoken soft_prompt; do
  run "evaluate: ${variant}" \
    "configs/v2_clegr_extended_exact2x_${variant}.yaml" \
    "v2_clegr_extended_exact2x_${variant}_eval" \
    evaluate
done

log "ALL DONE"
