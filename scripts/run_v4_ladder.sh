#!/usr/bin/env bash
# Frozen v4.1 GPU ladder (one Blackwell-class GPU, ~12-15 h budget).
#
# Usage (from the repo root on the GPU server):
#   LM_PATH=/path/to/Meta-Llama-3.1-8B-Instruct scripts/run_v4_ladder.sh            # all rungs
#   scripts/run_v4_ladder.sh setup soft gt42                                         # some rungs
#
# Rungs, in budget order (rough single-GPU estimates; real times go to $ART/timings.tsv):
#   setup   verify data hashes (regenerate if missing), Llama path, GPU, full pytest     ~15 min
#   soft    graph-blind soft-prompt control                                             ~1 h
#   gt42    GraphToken sum, seed 42                       <- go/no-go point (see checklist)  ~1.5 h
#   warm    contrastive GNN pretrain, then warm-started GraphToken (2 epochs)           ~3 h
#   gt43    GraphToken sum, seed 43                                                     ~1.5 h
#   gt44    GraphToken sum, seed 44                                                     ~1.5 h
#   random  frozen random sum GNN + projector (no-GNN-training control)                 ~1.5 h
#
# Each rung writes $ART/<rung>.done when it finishes; finished rungs are skipped on re-run, so
# the script can be restarted after a preemption. Delete the marker to force a re-run.
#
# Environment:
#   LM_PATH         local Llama-3.1-8B-Instruct dir; symlinked to the configs' model.lm_name if
#                   that path does not exist (configs stay untouched so provenance is clean)
#   ART             artifact dir copied after each rung (default: artifacts/v4_1)
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
    git rev-parse HEAD >"$dest/git_sha.txt"
    git status --porcelain >"$dest/git_status.txt"
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

do_setup() {
    nvidia-smi
    if [ "$GM" = "uv run graph-modi" ]; then uv sync --extra dev --extra tea; fi
    local lm
    lm=$($PY -c "import yaml; print(yaml.safe_load(open('configs/v4_graphtoken_s42.yaml'))['model']['lm_name'])")
    if [ ! -e "$lm" ]; then
        if [ -z "${LM_PATH:-}" ]; then
            log "ERROR: $lm missing; set LM_PATH to the local Llama-3.1-8B-Instruct dir"
            exit 1
        fi
        mkdir -p "$(dirname "$lm")"
        ln -s "$(realpath "$LM_PATH")" "$lm"
        log "linked $lm -> $LM_PATH"
    fi
    if [ ! -f "$DATA/static_train.jsonl" ]; then
        log "regenerating $DATA with scripts/audit_v4_gates.py (deterministic, seed 42)"
        PYTHONPATH=src $PY scripts/audit_v4_gates.py --work "$DATA" \
            --out "$ART/gates_regenerated" 2>&1 | tee "$ART/logs/regenerate.log"
        git checkout -- "$DATA/manifest.json" 2>/dev/null || true
    fi
    $PY - "$DATA" <<'EOF'
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
EOF
    $PY -m pytest -q 2>&1 | tee "$ART/logs/pytest.log"
    # The torch tests skip silently when torch is missing; on the GPU box none may be skipped.
    if grep -qE '[0-9]+ skipped' "$ART/logs/pytest.log"; then
        log "ERROR: tests were skipped on the GPU server (torch/transformers missing?)"
        exit 1
    fi
}

do_soft() {
    run soft_train train-projector --config configs/v4_soft_prompt.yaml
    for split in validation test; do
        run "soft_static_${split}" static-eval --config configs/v4_soft_prompt.yaml --split "$split" --checkpoint final
    done
    run soft_sessions evaluate --config configs/v4_soft_prompt.yaml --checkpoint final
    copy_artifacts outputs/v4_soft_prompt
}

graph_token_seed() {
    local seed=$1 config="configs/v4_graphtoken_s$1.yaml"
    run "gt${seed}_train" train-projector --config "$config"
    evaluate_graph_run "$config" "gt${seed}"
    copy_artifacts "outputs/v4_graphtoken_s${seed}"
}
do_gt42() { graph_token_seed 42; }
do_gt43() { graph_token_seed 43; }
do_gt44() { graph_token_seed 44; }

do_warm() {
    run pretrain_sum pretrain-gnn --config configs/v4_pretrain_sum.yaml
    copy_artifacts outputs/v4_pretrain_sum
    run warm_train train-projector --config configs/v4_graphtoken_warm_s42.yaml
    evaluate_graph_run configs/v4_graphtoken_warm_s42.yaml warm
    copy_artifacts outputs/v4_graphtoken_warm_s42
}

do_random() {
    run random_train train-projector --config configs/v4_tea_random_sum.yaml
    evaluate_graph_run configs/v4_tea_random_sum.yaml random
    copy_artifacts outputs/v4_tea_random_sum
}

rungs=("$@")
[ ${#rungs[@]} -gt 0 ] || rungs=(setup soft gt42 warm gt43 gt44 random)
for name in "${rungs[@]}"; do
    case $name in
        setup | soft | gt42 | gt43 | gt44 | warm | random) rung "$name" "do_$name" ;;
        *)
            echo "unknown rung: $name" >&2
            exit 2
            ;;
    esac
done
log "ladder finished: ${rungs[*]}; artifacts in $ART"
