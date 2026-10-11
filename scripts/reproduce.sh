#!/bin/bash
# Reproduce model_02a.pt from the raw recordings, then evaluate it.
# Run from the repository root:  bash scripts/reproduce.sh
# Needs data/raw/ prepared as described in README.md (public datasets + HackRF release bundles) and src/yolo26s.pt.
# Each step writes runs/logs/<step>.log and a marker runs/logs/<step>.done; finished steps are skipped when the script is run again.
# Optional environment variables:
#   PY        python interpreter (default: python)
#   WORKERS   processes for picture synthesis (default: 16)
#   AFTER     command run after each step, e.g. to free memory (default: none)
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}; WORKERS=${WORKERS:-16}; AFTER=${AFTER:-}
mkdir -p runs/logs runs/review doc/fig_report data/yaml   # some steps write here without creating the folder
step() {   # step <name> <command...>
  local name=$1; shift
  if [ -f runs/logs/$name.done ]; then echo "[skip] $name"; return 0; fi
  echo "[$(date '+%F %T')] $name: $*"
  if "$@" > runs/logs/$name.log 2>&1; then touch runs/logs/$name.done; echo "[$(date '+%F %T')] $name done"
  else echo "[$(date '+%F %T')] $name FAILED (runs/logs/$name.log)"; exit 1; fi
  [ -n "$AFTER" ] && eval "$AFTER" > /dev/null 2>&1 || true
}
neo2_pictures() {   # pictures of every Neo 2 HackRF session, used for automatic labels and test 4
  for d in data/raw/hackrf/3_neo2_park/[0-9][0-9]_* data/raw/hackrf/2_neo2_indoor/20261008_drone_*; do
    $PY deploy/src/iq_to_image.py --input "$d" --out "data/derived/hackrf/${d#data/raw/hackrf/}/images" || return 1
  done
}
drff_scan() {   # DRFF-R2: per-file occupancy (npz) is recomputed; the band candidates (json) and the human fixes come from data/labels/
  $PY src/drff_prep.py scan 6 || return 1
  cp -p data/labels/drff_scan_reviewed/*.json runs/review/drff_scan/ && cp -p data/labels/drff_band_fix.json runs/review/drff_band_fix.json
}
train_2a() {
  (cd src && $PY train_det.py --data ../data/yaml/v17_2a.yaml --model yolo26s.pt --out ../runs/det_v17_2a --epochs 40 --batch 16 --seed 0) || return 1
  cp -p runs/det_v17_2a/train/weights/last.pt runs/model_02a_reproduced.pt
}

# 1. Material
step 01_hackrf_clean      $PY src/hackrf_clean.py                                   # HackRF backgrounds with Bluetooth bursts removed
step 02_drff_scan         drff_scan                                                 # DRFF-R2: drone band and on-time per file (reviewed bands from data/labels/)
step 03_pieces_v3         $PY src/synth_iq_build.py pieces --ver v3                 # drone pieces (train/test) from DRFF-R2, Tampere, DroneDetect, RFUAV
step 04_pieces_v7         $PY src/synth_iq_build.py pieces_val --ver v7             # + validation pieces
step 05_neo2_pictures     neo2_pictures
step 06_neo2_autolabel    $PY src/neo2_autolabel.py --write                         # rule-based Neo 2 boxes
step 07_neo2_pieces       $PY src/neo2_pieces.py                                    # Neo 2 pieces, park background, noise shapes
# 2. Test sets 2 and 3, real "no drone" pictures
step 08_synth_v8          $PY src/synth_iq_build.py build --ver v8 --n_train 6000 --n_val 600 --n_test 1000 --workers $WORKERS
step 09_drff_v8_test      $PY src/drff_v8_test.py
step 10_lists_v13         $PY src/lists_v13.py
step 11_build_v14         $PY src/build_v14.py
# 3. Training pictures for model 2a and training
step 12_synth_v10         $PY src/synth_iq_build.py build --ver v10 --n_train 6000 --n_val 600 --n_test 0 --workers $WORKERS
step 13_build_v17         $PY src/build_v17.py
step 14_train_2a          train_2a
# 4. Tests 1-4 for the reproduced model and for the released model
step 15_eval_reproduced   $PY src/eval_v15.py --name m2a_reproduced --weights runs/model_02a_reproduced.pt
step 16_eval_released     $PY src/eval_v15.py --name m2a_released --weights deploy/models/model_02a.pt
step 17_eval_neo2         $PY src/eval_neo2_test.py --models m2a_reproduced=runs/model_02a_reproduced.pt m2a_released=deploy/models/model_02a.pt
echo "REPRODUCE_DONE"
