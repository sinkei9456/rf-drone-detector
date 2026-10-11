#!/bin/bash
# Download the 73 DRFF-R2 files used by this work (about 48 GB) into data/raw/drff_r2/.
# List: data/lists/drff_r2_files.txt (fileId size path). Resumable; a file counts as done when its size matches.
# Run from the repository root:  bash scripts/download_drff_r2.sh
cd "$(dirname "$0")/.."
OUT=data/raw/drff_r2; mkdir -p $OUT
grep -v '^#' data/lists/drff_r2_files.txt | grep 'dataset6-wifi_mixed/' > $OUT/dl_list.txt   # the code checks dataset6 files against this list
get() { id=$1; size=$2; out=data/raw/drff_r2/$3; mkdir -p "$(dirname "$out")"
  for try in 1 2 3 4 5; do
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    [ "$have" = "$size" ] && { echo "OK $out"; return 0; }
    curl -s -L -C - -m 3600 -o "$out" "https://china.scidb.cn/download?fileId=$id"
  done
  echo "FAIL $out $(stat -c %s "$out" 2>/dev/null)"; }
export -f get
grep -v '^#' data/lists/drff_r2_files.txt | xargs -P 4 -n 3 bash -c 'get "$0" "$1" "$2"' | tee $OUT/dl.log
echo "done: $(grep -c ^OK $OUT/dl.log) ok, $(grep -c ^FAIL $OUT/dl.log) failed"
