#!/bin/bash
# Run verifier-loop stages for one boundary on one GPU, writing a report after
# each stage; stops at the first failing stage.
#   scripts/run_loop.sh <iteration> <PR|BV> <gpu> [stages...]
# Environment: PY (python with torch/transformers), HF_HOME, GEOFLIP_WORK.
set -o pipefail
IT=$1; B=$2; GPU=$3; shift 3
STAGES=${*:-check ambiguity screen confirm test matrix}
cd "$(dirname "$0")/.."
PY=${PY:-python}
export CUDA_VISIBLE_DEVICES=$GPU
for S in $STAGES; do
  echo "== $S $(date +%T)"
  $PY -m geoflip.pipeline "$S" --iteration "$IT" --boundary "$B" 2>&1 \
    | grep -v "Loading weights\|clean_up\|^$" || exit 1
  $PY -m geoflip.pipeline report --iteration "$IT" --boundary "$B" > /dev/null
done
echo "== done $(date +%T)"
