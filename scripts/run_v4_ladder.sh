#!/usr/bin/env bash
# Frozen v4.1 GPU ladder (one Blackwell-class GPU, ~12-15 h budget).
#
# Usage (from the repo root on the GPU server):
#   LM_PATH=/path/to/Meta-Llama-3.1-8B-Instruct scripts/run_v4_ladder.sh            # all rungs
#   scripts/run_v4_ladder.sh setup soft gt42                                         # some rungs
#
# Rungs, in budget order (rough single-GPU estimates; real times go to $ART/timings.tsv):
#   setup   full pytest on the GPU box (no test may be skipped)                       ~15 min
#   soft    graph-blind soft-prompt control                                             ~1 h
#   gt42    GraphToken sum, seed 42                       <- go/no-go point (see checklist)  ~1.5 h
#   tea42   TEA-GLM GNN pretrain, then TEA-style (frozen pretrained GNN), seed 42      ~1.5 h
#   warm    warm-started GraphToken (pretrained GNN trained jointly, 1 epoch)           ~1 h
#   tea43   TEA-style, seed 43                                                          ~1 h
#   tea44   TEA-style, seed 44                                                          ~1 h
#   gt43    GraphToken sum, seed 43                                                     ~1.5 h
#   gt44    GraphToken sum, seed 44                                                     ~1.5 h
#   random  frozen random sum GNN + projector (no-GNN-training control)                 ~1.5 h
#   text    zero-shot serialized-graph text baselines (frozen Llama, needs gt42)        ~30 min
#   sweep   projector lr 3e-4 / 1e-4 for GraphToken s42, TEA-style s42, random GNN, warm start
#           (training + validation only; picks lr on validation loss)               ~1 h
#
# Every launch first runs a cheap preflight (uv present, torch sees the GPU, Llama path, data
# hashes, BERT download), even when setup.done exists, so a new VM is checked before any rung
# (SKIP_PREFLIGHT=1 turns it off). Each rung writes $ART/<rung>.done when it finishes; finished
# rungs are skipped on re-run, so the script can be restarted after a preemption. Delete the marker
# to force a re-run. Inside a rung, a training step whose checkpoint-final already exists is not
# repeated, so a crash during evaluation does not cost a retrain.
#
# Environment:
#   LM_PATH         local Llama-3.1-8B-Instruct dir; symlinked to the configs' model.lm_name if
#                   that path does not exist (configs stay untouched so provenance is clean)
#   ART             artifact dir copied after each rung (default: artifacts/v4_1; a relative
#                   path is relative to the repo root). Put it on persistent storage.
#   GM              CLI launcher (default: "uv run graph-modi")
#   COPY_CHECKPOINTS=1  also copy checkpoint-best / checkpoint-final dirs into $ART
set -euo pipefail

cd "$(dirname "$0")/.."
GM=${GM:-"uv run graph-modi"}
PY=${PY:-"uv run python"}
ART=${ART:-artifacts/v4_1}
DATA=datasets/metro_v4_1
mkdir -p "$ART/logs"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$ART/ladder.log"; }

run() {
    # run <logname> <cli args...>
    local name=$1
    shift
    log "RUN $name: $GM $*"
    $GM "$@" --no-progress 2>&1 | tee "$ART/logs/$name.log"
}

prune_checkpoints() {
    # Keep the checkpoint-best / checkpoint-final weights; drop the periodic resume checkpoints and
    # optimizer state (~6 GB each with the 4096-wide readout), which evaluation never reads.
    local out=$1 sub
    for sub in projector soft_prompt; do
        [ -d "$out/$sub" ] || continue
        find "$out/$sub" -maxdepth 1 -type d -name 'checkpoint-0*' -exec rm -rf {} +
        rm -f "$out/$sub"/checkpoint-*/optimizer.pt
    done
}

copy_artifacts() {
    local out=$1 dest="$ART/$(basename "$1")"
    mkdir -p "$dest"
    find "$out" -maxdepth 1 -type f \( -name '*.json' -o -name '*.md' -o -name '*.jsonl' \) \
        -exec cp {} "$dest/" \;
    for sub in projector soft_prompt gnn; do
        [ -d "$out/$sub" ] || continue
        mkdir -p "$dest/$sub"
        find "$out/$sub" -maxdepth 1 -type f -name '*.json' -exec cp {} "$dest/$sub/" \;
        for ckpt in "$out/$sub"/checkpoint-best "$out/$sub"/checkpoint-final; do
            [ -d "$ckpt" ] || continue
            mkdir -p "$dest/$sub/$(basename "$ckpt")"
            find "$ckpt" -maxdepth 1 -type f -name '*.json' -exec cp {} "$dest/$sub/$(basename "$ckpt")/" \;
            if [ "${COPY_CHECKPOINTS:-0}" = 1 ]; then cp -r "$ckpt" "$dest/$sub/"; fi
        done
        if [ "$sub" = gnn ] && [ "${COPY_CHECKPOINTS:-0}" = 1 ]; then cp "$out/gnn/"*.pt "$dest/gnn/" 2>/dev/null || true; fi
    done
    git rev-parse HEAD >"$dest/git_sha.txt" 2>/dev/null || echo unknown >"$dest/git_sha.txt"
    git status --porcelain >"$dest/git_status.txt" 2>/dev/null || echo unknown >"$dest/git_status.txt"
    prune_checkpoints "$out"
}

rung() {
    # rung <name> <function>
    local name=$1 fn=$2
    if [ -f "$ART/$name.done" ]; then
        log "SKIP $name (done)"
        return
    fi
    local start=$SECONDS
    log "START $name"
    "$fn"
    local secs=$((SECONDS - start))
    printf '%s\t%s\t%s\n' "$name" "$secs" "$(date '+%F %T')" >>"$ART/timings.tsv"
    touch "$ART/$name.done"
    log "DONE $name in $((secs / 60)) min"
}

# Static eval of best and final on validation + test, then multi-turn sessions (test) on best.
evaluate_graph_run() {
    local config=$1 tag=$2
    for ckpt in best final; do
        for split in validation test; do
            run "${tag}_static_${split}_${ckpt}" static-eval --config "$config" --split "$split" --checkpoint "$ckpt"
        done
    done
    run "${tag}_sessions_best" evaluate --config "$config" --checkpoint best
}

train_once() {
    # train_once <logname> <done-file> <cli args...>: skip the step when its output already exists
    local name=$1 done_file=$2
    shift 2
    if [ -f "$done_file" ]; then
        log "SKIP $name ($done_file exists)"
        return
    fi
    run "$name" "$@"
}

preflight() {
    if [ "$GM" = "uv run graph-modi" ] && ! command -v uv >/dev/null; then
        log "ERROR: uv not found; install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    # Locked sync with the training extras (a no-op when already synced); torch comes from here.
    if [ "$GM" = "uv run graph-modi" ]; then uv sync --locked --extra dev --extra tea; fi
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a "$ART/ladder.log"
    # torch must actually reach the GPU; otherwise every rung would silently train on CPU.
    $PY - <<'PYEOF' 2>&1 | tee -a "$ART/ladder.log"
import sys
import torch
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("ERROR: torch cannot use the GPU. The locked torch wheel may need a newer NVIDIA "
             "driver than this machine has; see documents/experiments/v4_run_checklist.md")
x = torch.ones(1024, 1024, device="cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}, matmul ok: {float((x @ x)[0, 0]) == 1024.0}")
PYEOF
    local lm
    lm=$($PY -c "import yaml; print(yaml.safe_load(open('configs/v4_graphtoken_s42.yaml'))['model']['lm_name'])")
    if [ ! -e "$lm" ]; then
        if [ -z "${LM_PATH:-}" ]; then
            log "ERROR: $lm missing; set LM_PATH to the local Llama-3.1-8B-Instruct dir"
            exit 1
        fi
        if ! mkdir -p "$(dirname "$lm")" 2>/dev/null; then
            log "ERROR: cannot create $(dirname "$lm"); run as root or create it with sudo"
            exit 1
        fi
        ln -s "$(realpath "$LM_PATH")" "$lm"
        log "linked $lm -> $LM_PATH"
    fi
    if [ ! -f "$lm/config.json" ] || [ ! -f "$lm/model.safetensors.index.json" ]; then
        log "ERROR: $lm needs config.json and model.safetensors.index.json (HF safetensors layout)"
        exit 1
    fi
    if [ ! -f "$DATA/static_train.jsonl" ]; then
        log "regenerating $DATA with scripts/audit_v4_gates.py (deterministic, seed 42)"
        PYTHONPATH=src $PY scripts/audit_v4_gates.py --work "$DATA" \
            --out "$ART/gates_regenerated" 2>&1 | tee "$ART/logs/regenerate.log"
        git checkout -- "$DATA/manifest.json" 2>/dev/null || true
    fi
    $PY - "$DATA" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
bad = []
for name, entry in manifest["files"].items():
    digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        bad.append(name)
    print(f"{name}: {'ok' if digest == entry['sha256'] else 'MISMATCH'}")
if bad:
    sys.exit(f"sha256 mismatch: {bad}; do not train on this data")
PYEOF
    # Fail now, not mid-rung, if the BERT node encoder cannot be downloaded or loaded.
    $PY -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('bert-base-uncased'); AutoModel.from_pretrained('bert-base-uncased'); print('bert-base-uncased ok')"
}

do_setup() {
    # Override pyproject's addopts so a single -q still prints the "N passed, M skipped" summary,
    # list skip reasons (-rs), and keep pytest's cache out of the repo (git_dirty stays false).
    $PY -m pytest -o addopts="" -q -rs -p no:cacheprovider 2>&1 | tee "$ART/logs/pytest.log"
    # The torch tests skip silently when torch is missing; on the GPU box none may be skipped.
    if grep -qE '[0-9]+ skipped' "$ART/logs/pytest.log"; then
        log "ERROR: tests were skipped on the GPU server (torch/transformers/accelerate missing?)"
        exit 1
    fi
    if ! grep -qE '[0-9]+ passed' "$ART/logs/pytest.log"; then
        log "ERROR: no pytest summary line in $ART/logs/pytest.log"
        exit 1
    fi
}

do_soft() {
    train_once soft_train outputs/v4_soft_prompt/soft_prompt/checkpoint-final/metadata.json \
        train-projector --config configs/v4_soft_prompt.yaml
    for split in validation test; do
        run "soft_static_${split}" static-eval --config configs/v4_soft_prompt.yaml --split "$split" --checkpoint final
    done
    run soft_sessions evaluate --config configs/v4_soft_prompt.yaml --checkpoint final
    copy_artifacts outputs/v4_soft_prompt
}

graph_token_seed() {
    local seed=$1 config="configs/v4_graphtoken_s$1.yaml"
    train_once "gt${seed}_train" "outputs/v4_graphtoken_s${seed}/projector/checkpoint-final/metadata.json" \
        train-projector --config "$config"
    evaluate_graph_run "$config" "gt${seed}"
    copy_artifacts "outputs/v4_graphtoken_s${seed}"
}
do_gt42() { graph_token_seed 42; }
do_gt43() { graph_token_seed 43; }
do_gt44() { graph_token_seed 44; }

pretrain_once() {
    train_once pretrain_sum outputs/v4_pretrain_sum/gnn/metadata.json \
        pretrain-gnn --config configs/v4_pretrain_sum.yaml
    copy_artifacts outputs/v4_pretrain_sum
}

tea_seed() {
    local seed=$1 config="configs/v4_tea_s$1.yaml"
    pretrain_once
    train_once "tea${seed}_train" "outputs/v4_tea_s${seed}/projector/checkpoint-final/metadata.json" \
        train-projector --config "$config"
    evaluate_graph_run "$config" "tea${seed}"
    copy_artifacts "outputs/v4_tea_s${seed}"
}
do_tea42() { tea_seed 42; }
do_tea43() { tea_seed 43; }
do_tea44() { tea_seed 44; }

do_warm() {
    pretrain_once
    train_once warm_train outputs/v4_graphtoken_warm_s42/projector/checkpoint-final/metadata.json \
        train-projector --config configs/v4_graphtoken_warm_s42.yaml
    evaluate_graph_run configs/v4_graphtoken_warm_s42.yaml warm
    copy_artifacts outputs/v4_graphtoken_warm_s42
}

do_random() {
    train_once random_train outputs/v4_tea_random_sum/projector/checkpoint-final/metadata.json \
        train-projector --config configs/v4_tea_random_sum.yaml
    evaluate_graph_run configs/v4_tea_random_sum.yaml random
    copy_artifacts outputs/v4_tea_random_sum
}

do_text() {
    run text_sessions evaluate --config configs/v4_text_baselines.yaml
    copy_artifacts outputs/v4_text_baselines
}

do_sweep() {
    # Training and validation loss only; checkpoints are deleted, the val histories are kept.
    pretrain_once
    local config name
    for config in configs/sweep/*.yaml; do
        name=$(basename "$config" .yaml)
        train_once "sweep_${name}" "outputs/sweep/${name}/projector/checkpoint-final/metadata.json" \
            train-projector --config "$config"
        mkdir -p "$ART/sweep/$name/projector"
        cp "outputs/sweep/${name}/projector/"*.json "$ART/sweep/$name/projector/"
        rm -rf "outputs/sweep/${name}/projector/"checkpoint-0* \
            "outputs/sweep/${name}/projector/checkpoint-best"/*.pt
        rm -f "outputs/sweep/${name}/projector/checkpoint-final"/{gnn,projector,optimizer}.pt
    done
    $PY scripts/sweep_projector_lr.py --runs outputs --out "$ART/projector_lr_sweep.json" |
        tee -a "$ART/ladder.log"
}

rungs=("$@")
[ ${#rungs[@]} -gt 0 ] || rungs=(setup soft gt42 tea42 warm gt43 gt44 tea43 tea44 random text)
if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
    log "preflight"
    preflight
fi
for name in "${rungs[@]}"; do
    case $name in
        setup | soft | gt42 | gt43 | gt44 | tea42 | tea43 | tea44 | warm | random | text | sweep) rung "$name" "do_$name" ;;
        *)
            echo "unknown rung: $name" >&2
            exit 2
            ;;
    esac
done
log "ladder finished: ${rungs[*]}; artifacts in $ART"
