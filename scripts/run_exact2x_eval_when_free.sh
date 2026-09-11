#!/usr/bin/env bash
# Wait for enough free GPU memory, then run exact2x TEA + GraphToken (+ soft_prompt).
set -euo pipefail
cd /home/arihantr/ANLP/Graph-Modi
PY=/home/arihantr/ANLP/.venv/bin/python
NEED_MIB=20000

free_mib() {
  local idx="$1"
  nvidia-smi -i "$idx" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

echo "[exact2x-eval] waiting for GPU0 and GPU1 to each have >= ${NEED_MIB} MiB free..."
while true; do
  f0=$(free_mib 0)
  f1=$(free_mib 1)
  echo "[exact2x-eval] free MiB: GPU0=${f0} GPU1=${f1} $(date -Is)"
  if [[ "$f0" -ge "$NEED_MIB" && "$f1" -ge "$NEED_MIB" ]]; then
    break
  fi
  sleep 60
done

echo "[exact2x-eval] launching TEA on GPU0 and GraphToken on GPU1"
CUDA_VISIBLE_DEVICES=0 "$PY" -m graph_modi.cli evaluate --config configs/v2_gate_variant_cf_exact2x_tea.yaml \
  2>&1 | tee outputs/v2_gate_variant_cf_exact2x_tea_eval.log &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "$PY" -m graph_modi.cli evaluate --config configs/v2_gate_variant_cf_exact2x_graphtoken.yaml \
  2>&1 | tee outputs/v2_gate_variant_cf_exact2x_graphtoken_eval.log &
PID1=$!
wait $PID0
echo "[exact2x-eval] TEA exit $?"
wait $PID1
echo "[exact2x-eval] GraphToken exit $?"

echo "[exact2x-eval] launching soft_prompt on GPU0"
CUDA_VISIBLE_DEVICES=0 "$PY" -m graph_modi.cli evaluate --config configs/v2_gate_variant_cf_exact2x_soft_prompt.yaml \
  2>&1 | tee outputs/v2_gate_variant_cf_exact2x_soft_prompt_eval.log
echo "[exact2x-eval] soft_prompt done"
echo "[exact2x-eval] ALL DONE $(date -Is)"
