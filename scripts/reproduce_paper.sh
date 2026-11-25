#!/bin/bash
# Reproduce the experiments of the paper (v2) from the raw IQ data.
# Run from the repository root:  bash scripts/reproduce_paper.sh [part ...]
#   parts: materials  generated  crowded  figures   (default: materials generated crowded)
#   materials : park background, drone signal segments (steps 02-07 of the hackathon pipeline, v1/scripts/reproduce.sh; its step 01, the receiver noise, is not used)
#   generated : models trained with generated interference: 4 densities x 5 folds x 2 seeds = 40 models (about 38 h on one DGX Spark)
#   crowded   : evaluation on the crowded sites and the real-background models (needs the crowded-site IQ data, which is not public)
#   figures   : figures of the paper (needs the NanumGothic font for the Korean labels)
# Needs data/raw/ prepared as described in README.md and the COCO-pretrained src/yolo26s.pt (GitHub release v1.0).
# Each step writes runs/logs/paper_<step>.log and a marker runs/logs/paper_<step>.done; finished steps are skipped when the script is run again.
# Optional environment variables:
#   PY        python interpreter (default: python)
#   WORKERS   processes for image synthesis (default: 16)
#   AFTER     command run after each step, e.g. to free memory (default: none)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}; WORKERS=${WORKERS:-16}; AFTER=${AFTER:-}
R=paper/runs/rev1; X=$R/exp2
mkdir -p runs/logs runs/review doc/fig_report $R/models $X/models   # some steps write here without creating the folder
step() {   # step <name> <command...>
  local name=paper_$1; shift
  if [ -f runs/logs/$name.done ]; then echo "[skip] $name"; return 0; fi
  echo "[$(date '+%F %T')] $name: $*"
  if "$@" > runs/logs/$name.log 2>&1; then touch runs/logs/$name.done; echo "[$(date '+%F %T')] $name done"
  else echo "[$(date '+%F %T')] $name FAILED (runs/logs/$name.log)"; exit 1; fi
  [ -n "$AFTER" ] && eval "$AFTER" > /dev/null 2>&1 || true
}
expect() {   # expect <marker> <command...>: run, then fail unless the output has the marker (checks that print a marker instead of failing)
  local marker=$1 out; shift; out=$("$@") || { echo "$out"; return 1; }; echo "$out"; grep -q "$marker" <<< "$out"
}
pictures() {   # pictures <input folder> ...: 20 ms spectrogram images of every IQ file, the same transform as training
  for d in "$@"; do $PY src/iq_to_image.py --input "$d" --out "data/derived/hackrf/${d#data/raw/hackrf/}/images" || return 1; done
}
drff_scan() {   # DRFF-R2: per-file occupancy is recomputed; the band candidates and the human fixes come from data/labels/
  $PY src/drff_prep.py scan 6 || return 1
  cp -p data/labels/drff_scan_reviewed/*.json runs/review/drff_scan/ && cp -p data/labels/drff_band_fix.json runs/review/drff_band_fix.json
}
train() {   # train <runs folder> <name> <seed>: YOLO26s from COCO weights, 40 epochs, batch 16, last epoch weights
  local out=$1 n=$2 seed=$3
  (cd src && $PY train_det.py --data ../$out/yaml/$n.yaml --model yolo26s.pt --out ../$out/det_$n --epochs 40 --batch 16 --seed $seed) || return 1
  cp -p $out/det_$n/train/weights/last.pt $out/models/$n.pt
}
eval_generated() { expect "REV1_EVAL_DONE $1" $PY -W ignore paper/src/rev1_eval.py --name $1 --weights $R/models/$1.pt; }

materials() {
  step 02_drff_scan      drff_scan                                        # DRFF-R2: drone band and on-time per file
  step 03_pieces_v3      $PY src/synth_iq_build.py pieces --ver v3        # drone signal segments (train/test) from DRFF-R2, Tampere, DroneDetect, RFUAV
  step 04_pieces_v7      $PY src/synth_iq_build.py pieces_val --ver v7    # + validation segments -> runs/iq_pieces_v7.npy (303 segments, 20 models)
  step 05_neo2_pictures  pictures data/raw/hackrf/3_neo2_park/0[1-9]_* data/raw/hackrf/3_neo2_park/1[0-3]_* data/raw/hackrf/2_neo2_indoor/20261008_drone_*
  step 06_neo2_autolabel $PY src/neo2_autolabel.py --write                # rule-based Neo 2 boxes (labels of the low-interference real test)
  step 07_neo2_pieces    $PY src/neo2_pieces.py                           # park background with short bursts removed, Neo 2 segments
}

generated() {
  step 10_calib        $PY paper/src/rev1_synth.py calib                  # burst counts that give 2/20/50/80 % median occupancy
  step 11_synth        $PY paper/src/rev1_synth.py build --workers $WORKERS
  step 12_summary      $PY paper/src/rev1_synth.py summary
  step 14_test1_2      expect REV1_T2_BUILD_DONE $PY paper/src/rev1_test1_2.py build --workers $WORKERS
  step 15_leak         expect REV1_LEAK_OK $PY paper/src/rev1_leak.py     # data leakage check; training starts only if every check passes
  for seed in 1 2; do
    for fold in f1_mini f2_air f3_mavic f4_pro f5_other; do
      for cond in c02 c20 c50 c80; do
        n=${fold}_${cond}_s${seed}
        step 20_train_$n train $R $n $seed
        step 21_eval_$n  eval_generated $n                               # validation threshold, unseen-family test, low-interference real test
      done
    done
  done
  step 22_test1_2_eval $PY -W ignore paper/src/rev1_test1_2.py eval        # Neo 2 synthetic test
  step 23_report       $PY -W ignore paper/src/rev1_report.py              # -> paper/runs/rev1/RESULTS.md, results.json
}

crowded() {
  step 30_crowded_pictures pictures data/raw/hackrf/4_neo2_crowded_20261114/01_수원역_플랫폼 data/raw/hackrf/4_neo2_crowded_20261114/02_열차_객실
  step 31_test1_4_clips  $PY paper/src/rev1_test1_4.py clips               # bursts and occupancy per IQ file
  step 32_test1_4_eval   $PY paper/src/rev1_test1_4.py eval                # the 50 models on the crowded sites
  step 33_test1_4_report $PY paper/src/rev1_test1_4.py report
  step 34_realbg_build   expect EXP2_LEAK_OK $PY -W ignore paper/src/rev1_exp2.py build --workers $WORKERS
  for seed in 1 2; do
    for k in test2_1 test2_2; do
      n=${k}_s${seed}
      step 35_train_$n train $X $n $seed
      step 36_eval_$n  $PY -W ignore paper/src/rev1_exp2.py eval --only $n
    done
  done
  step 37_realbg_report  $PY -W ignore paper/src/rev1_exp2.py report       # -> paper/runs/rev1/exp2/RESULTS_exp2.md, results_exp2.json
}

figures() {
  mkdir -p paper/fig
  step 40_figures  $PY paper/src/fig_rev2.py
  step 41_curves   $PY paper/src/fig_exp2_curves.py --rev2
}

for part in ${@:-materials generated crowded}; do
  case $part in
    materials|generated|crowded|figures) $part ;;
    *) echo "unknown part: $part (materials, generated, crowded, figures)"; exit 1 ;;
  esac
done
echo "REPRODUCE_PAPER_DONE"
