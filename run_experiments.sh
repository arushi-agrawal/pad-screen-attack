#!/bin/bash
# Every experiment in the report, in the order it was run. Sequential on purpose: one 4-thread CPU job at a time.
# Usage: ./run_experiments.sh [stage]   stages: splits baseline cnn density facerel anomaly analysis freeze ship all
set -e; cd "$(dirname "$0")"; PY=${PY:-python}; mkdir -p results/logs
log() { echo "=== $1  $(date +%H:%M:%S)"; }
stage=${1:-all}
run_if() { [ "$stage" = all ] || [ "$stage" = "$1" ]; }
run_if splits    && { log splits;   $PY scripts/make_splits.py; $PY pad/data.py; }
run_if baseline  && { log baseline; $PY scripts/baselines.py | tee results/logs/baselines.txt; }
run_if cnn       && { log cnn
  $PY -u scripts/train.py --aug none                         > results/logs/patchcnn_none_devcv.txt 2>&1
  $PY -u scripts/train.py --aug full                         > results/logs/patchcnn_full_devcv.txt 2>&1
  $PY -u scripts/train.py --aug full --protocol lopo         > results/logs/patchcnn_full_lopo.txt 2>&1
  $PY -u scripts/train.py --aug full --resolution facenorm   > results/logs/patchcnn_full_devcv_facenorm.txt 2>&1
  $PY -u scripts/train.py --aug full --resolution facenorm --protocol lopo > results/logs/patchcnn_full_lopo_facenorm.txt 2>&1
  $PY -u scripts/train.py --aug wide                         > results/logs/patchcnn_wide_devcv.txt 2>&1
  $PY -u scripts/train.py --aug wide --protocol lopo         > results/logs/patchcnn_wide_lopo.txt 2>&1; }
run_if density   && { log density
  $PY -u scripts/train.py --aug none --eval density --eval_only                 > results/logs/patchcnn_none_devcv_density.txt 2>&1
  $PY -u scripts/train.py --aug full --eval density --eval_only                 > results/logs/patchcnn_full_devcv_density.txt 2>&1
  $PY -u scripts/train.py --aug wide --eval density --eval_only                 > results/logs/patchcnn_wide_devcv_density.txt 2>&1
  $PY -u scripts/train.py --aug full --protocol lopo --eval density --eval_only > results/logs/patchcnn_full_lopo_density.txt 2>&1
  $PY -u scripts/train.py --aug wide --protocol lopo --eval density --eval_only > results/logs/patchcnn_wide_lopo_density.txt 2>&1; }
run_if facerel   && { log facerel
  $PY -u scripts/train.py --aug facerel                              > results/logs/patchcnn_facerel_devcv.txt 2>&1
  $PY -u scripts/train.py --aug facerel --eval density --eval_only   > results/logs/patchcnn_facerel_devcv_density.txt 2>&1
  $PY -u scripts/train.py --aug facerel --protocol lopo              > results/logs/patchcnn_facerel_lopo.txt 2>&1; }
run_if anomaly   && { log anomaly;  $PY scripts/anomaly.py        > results/logs/anomaly.txt 2>&1; }
run_if analysis  && { log analysis; $PY scripts/error_analysis.py > results/logs/error_analysis.txt 2>&1; $PY scripts/tradeoffs.py > results/logs/tradeoffs.txt 2>&1; $PY scripts/plot_phone_transfer.py; }
run_if freeze    && { log freeze;   $PY -u scripts/train.py --aug wide --protocol final > results/logs/final.txt 2>&1; }
run_if ship      && { log ship;     $PY -u scripts/train.py --aug wide --protocol ship  > results/logs/ship.txt 2>&1; }
log "done ($stage)"
