"""드론만 찾는 모델(10-07 저녁, 사용자 승인 "클래스를 드론에 한정, 지금 하는 것 끝내고 돌리자", 보고서 report_261008)의 학습 자료 만들기.
1) 합성 그림(잘린 신호 포함)은 그대로 쓰고 라벨에서 drone 박스만 남긴다(wifi·bt 박스 지움) → data/derived/yolo_syn_iq_v8_drone/
2) "드론 아님" 실제 그림(박스 없음) → data/derived/yolo_neg_v14/
   - DRFF-R2 학습용·검증용 드론 녹음(테스트용 녹음은 안 씀)에서 드론 영상 신호가 없는 주파수: 중심 2.41·2.42·2.43 GHz(그림 범위 2.40~2.44 GHz,
     드론 신호 범위 2.442~2.464 GHz 와 2 MHz 이상 떨어짐). 실제 와이파이·블루투스와 그 수신기 바탕이 들어 있다.
     학습용 녹음은 시각 15곳(0.05 + 0.088·k 초), 검증용 녹음은 5곳(0.05 + 0.27·k 초).
   - HackRF 안테나 뺀 녹음(블루투스를 지우지 않은 그대로): 첫째·둘째 녹음 50장씩 → 학습, 셋째 녹음 50장 → 검증. 넷째·다섯째와 안테나 끼운 녹음은 테스트에만.
   그림은 synth_iq.spectrogram_v7(가운데 줄 지움).
3) 목록·yaml(종류 1개: drone): data/derived/split_v7/{a_train,a_val,a_syn_test}.txt, data/yaml/v14_a.yaml, v14_test_syn.yaml, v14_test_drff.yaml
출력 요약: runs/review/build_v14.txt"""
import sys, os, glob
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import sys as _sys; _sys.path.insert(0, str(ROOT / "src")); import hackrf_paths as HP   # 10-09 HackRF 폴더 정리: 녹음 경로는 hackrf_paths 에서
import drff_prep as DP
import synth_iq as S
from spectrogram import to_image
SYN = ROOT / "data/derived/yolo_syn_iq_v8"; SD = ROOT / "data/derived/yolo_syn_iq_v8_drone"; NEG = ROOT / "data/derived/yolo_neg_v14"; SP = ROOT / "data/derived/split_v7"
for d in (SD / "images", SD / "labels", NEG / "images", NEG / "labels", SP): d.mkdir(parents=True, exist_ok=True)
CENTERS = (2.41e9, 2.42e9, 2.43e9)


def drff_job(a):
    stem, part = a
    f = next(DP.RAW.rglob(f"{stem}.mat")); x, fc = DP.load(f); out = []
    t0s = [0.05 + 0.088 * k for k in range(15)] if part == "train" else [0.05 + 0.27 * k for k in range(5)]
    for c in CENTERS:
        for k, t0 in enumerate(t0s):
            s0 = int(t0 * DP.FS_IN) - 20000; seg = x[s0: s0 + int(DP.WIN * DP.FS_IN) + 40000]; tt = (np.arange(len(seg)) + s0) / DP.FS_IN
            y = resample_poly((seg * np.exp(-2j * np.pi * (c - fc) * tt)).astype(np.complex64), 1, 5).astype(np.complex64)[4000: 4000 + DP.N]
            name = f"neg_drff_{stem}_{int(round(c / 1e6))}_t{k:02d}"
            to_image(S.spectrogram_v7(y)).save(NEG / "images" / f"{name}.png"); (NEG / "labels" / f"{name}.txt").write_text("")
            out.append((part, str((NEG / "images" / f"{name}.png").resolve())))
    return out


if __name__ == "__main__":
    L = []
    # 1) 합성: drone 박스만
    rd = lambda p: [l.strip() for l in open(p) if l.strip()]
    syn = {sp: [] for sp in ("train", "val", "test")}; nb = {"drone": 0, "removed": 0}
    for sp in syn:
        for p in rd(SYN / f"{sp}.txt"):
            p = Path(p); li = SD / "images" / p.name
            if not li.exists(): os.symlink(p, li)
            lines = [l for l in open(SYN / "labels" / f"{p.stem}.txt") if l.strip()]
            keep = [l for l in lines if l.split()[0] == "0"]; nb["drone"] += len(keep); nb["removed"] += len(lines) - len(keep)
            (SD / "labels" / f"{p.stem}.txt").write_text("".join(keep)); syn[sp].append(str(li))
    L.append(f"합성 그림: 학습 {len(syn['train'])} / 검증 {len(syn['val'])} / 테스트 {len(syn['test'])}장, drone 박스 {nb['drone']}개 남김, wifi·bt 박스 {nb['removed']}개 지움")
    # 2) 드론 아님 실제 그림
    EX = set(DP.excluded()); jobs = [(f.stem, DP.split_of(f.stem)) for f in DP.files() if f.stem not in EX and not f.stem.startswith("indoor") and DP.split_of(f.stem) in ("train", "val")]
    with Pool(4) as pool: res = [r for rr in pool.map(drff_job, jobs, chunksize=1) for r in rr]
    neg = {"train": [p for sp, p in res if sp == "train"], "val": [p for sp, p in res if sp == "val"]}
    L.append(f"DRFF-R2 드론 아님 그림: 학습용 녹음 {sum(1 for _, s in jobs if s == 'train')}개 → {len(neg['train'])}장, 검증용 녹음 {sum(1 for _, s in jobs if s == 'val')}개 → {len(neg['val'])}장 (중심 2.41·2.42·2.43 GHz)")
    hk = {"train": [], "val": []}
    for fid, sp in (("01", "train"), ("02", "train"), ("03", "val")):
        f = str(HP.nodrone_file("AntennaOff", fid))
        r = np.fromfile(f, dtype=np.int8).astype(np.float32); x = (r[0::2] + 1j * r[1::2]).astype(np.complex64)
        for i in range(len(x) // S.N):
            name = f"neg_hackrf_off{fid}_f{i:02d}"; to_image(S.spectrogram_v7(x[i * S.N:(i + 1) * S.N])).save(NEG / "images" / f"{name}.png")
            (NEG / "labels" / f"{name}.txt").write_text(""); hk[sp].append(str((NEG / "images" / f"{name}.png").resolve()))
    L.append(f"HackRF 안테나 뺀 녹음(지우지 않음) 드론 아님 그림: 학습 {len(hk['train'])}장(첫째·둘째), 검증 {len(hk['val'])}장(셋째)")
    # 3) 목록·yaml
    out = {"a_train": syn["train"] + neg["train"] + hk["train"], "a_val": syn["val"] + neg["val"] + hk["val"], "a_syn_test": syn["test"]}
    for k, v in out.items(): (SP / f"{k}.txt").write_text("\n".join(v) + "\n"); L.append(f"{k}: {len(v)}장")
    Y = "path: " + str(ROOT / "data/derived") + "\ntrain: {tr}\nval: {va}\nnames:\n  0: drone\n"
    (ROOT / "data/yaml/v14_a.yaml").write_text(Y.format(tr="split_v7/a_train.txt", va="split_v7/a_val.txt"))
    (ROOT / "data/yaml/v14_test_syn.yaml").write_text(Y.format(tr="split_v7/a_train.txt", va="split_v7/a_syn_test.txt"))       # 평가(val)에만
    (ROOT / "data/yaml/v14_test_drff.yaml").write_text(Y.format(tr="split_v7/a_train.txt", va="split_v6/drff8_main.txt"))     # 평가(val)에만
    (ROOT / "runs/review/build_v14.txt").write_text("\n".join(L) + "\n"); print("\n".join(L)); print("BUILD_V14_DONE")
