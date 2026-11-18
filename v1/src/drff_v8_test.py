"""DRFF-R2 실제 테스트셋 — 20 MHz 그림에 드론 신호가 보이는 비율별 (2026-10-07, 새 분석: 학습·검증은 합성만, 테스트는 실제 데이터를 20 MHz 로 가공).
DRFF-R2 테스트용 드론 녹음 20개(여러 대 기종의 넷째 드론, Mini 3 Pro 둘째 드론, 한 대 기종의 비행 방식 c6) + 드론 없는 실내 녹음 1개.
녹음은 100 MS/s(중심 2.437 GHz)라 중심 주파수를 마음대로 정해 20 MHz 그림(가로 20 ms)을 다시 만들 수 있다.
  드론 그림: 녹음마다 드론 신호 범위(사람 확인, w = 18~19 MHz, Air 2 는 9~10 MHz)가 그림 안에 v·w 만큼 보이게 중심을 정한다.
            v = 1.0(신호를 그림 가운데에), 0.9, 0.8, 0.7, 0.6, 0.5, 그리고 학습 기준(0.5) 아래인 0.3(참고). v<1 은 위쪽을 자르는 것과 아래쪽을
            자르는 것을 번갈아(녹음 순서 + v 순서).
  드론 없는 그림: 중심 2.41·2.43 GHz(드론 신호와 2 MHz 이상 떨어짐, 실내 와이파이가 있음). 드론 없는 실내 녹음은 2.41·2.43·2.45·2.47 GHz.
  시각: 녹음마다 0.005 + 0.14·k 초(k=0~9)에서 20 ms 씩 10장(원래 DRFF-R2 그림과 같은 시각).
그림: synth_iq.spectrogram_v7(가운데 줄 지움). 라벨: drone 박스만 = 그림 안에 보이는 드론 신호 범위 × 그 범위 안 신호 덩어리의 처음~끝
      (drff_prep.label_from_bursts 와 같은 방법; 보이는 폭이 4 MHz 미만이면 박스 없음). 와이파이·블루투스 박스는 없다(재지 않음).
출력: data/derived/yolo_drff_v8test/{images,labels}, data/derived/split_v6/drff8_{all,v100,v90,v80,v70,v60,v50,v30,nodrone}.txt, runs/review/drff_v8_test.csv"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import drff_prep as DP
import synth_iq as S
import linklabel
from spectrogram import to_image
OUT = ROOT / "data/derived/yolo_drff_v8test"; (OUT / "images").mkdir(parents=True, exist_ok=True); (OUT / "labels").mkdir(exist_ok=True)
SP = ROOT / "data/derived/split_v6"; SP.mkdir(exist_ok=True)
VS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3]
EX = set(DP.excluded()); B = DP.bands()
recs = sorted(f for f in DP.files() if DP.split_of(f.stem) == "test" and f.stem not in EX)
rows = []
for ri, f in enumerate(recs):
    x, fc = DP.load(f); st = f.stem; band = B.get(st)
    jobs = []
    if band is not None:
        b0, b1 = band; w = b1 - b0
        for vi, v in enumerate(VS):
            if v >= 1.0: c, side = (b0 + b1) / 2, "center"
            elif (ri + vi) % 2 == 0: c, side = b0 + v * w - S.FS / 2, "top_cut"
            else: c, side = b1 - v * w + S.FS / 2, "bottom_cut"
            jobs.append((f"v{int(v * 100)}", c, side))
        jobs += [("nodrone", 2.41e9, "-"), ("nodrone", 2.43e9, "-")]
    else:
        jobs += [("nodrone", c, "-") for c in (2.41e9, 2.43e9, 2.45e9, 2.47e9)]
    for grp, c, side in jobs:
        bw = None
        if band is not None:
            lo, hi = max(band[0], c - S.FS / 2) - c, min(band[1], c + S.FS / 2) - c; bw = (lo, hi) if hi > lo else None
        for k, t0 in enumerate(DP.T0S):
            s0 = int(t0 * DP.FS_IN) - 20000; seg = x[s0: s0 + int(DP.WIN * DP.FS_IN) + 40000]; tt = (np.arange(len(seg)) + s0) / DP.FS_IN
            y = resample_poly((seg * np.exp(-2j * np.pi * (c - fc) * tt)).astype(np.complex64), 1, 5).astype(np.complex64)[4000: 4000 + DP.N]
            Sdb = S.spectrogram_v7(y); P = 10 ** (Sdb.astype(np.float64) / 10)
            name = f"drff8_{st}_{grp}_{int(round(c / 1e5))}_w{k}"
            to_image(Sdb).save(OUT / "images" / f"{name}.png")
            b, _ = linklabel.detect_bursts(P, S.FS / P.shape[0], S.WIN / P.shape[1], f_min=-S.FS / 2)
            links, _ = DP.label_from_bursts([tuple(q) for q in b], bw, S.FS)
            lines = linklabel.links_to_yolo([L for L in links if L["cls"] == 0], S.FS, S.WIN, f_min=-S.FS / 2)
            with open(OUT / "labels" / f"{name}.txt", "w") as fh:
                for cl, xc, yc, ww, hh in lines: fh.write(f"{cl} {xc:.5f} {yc:.5f} {ww:.5f} {hh:.5f}\n")
            vis = 0.0 if bw is None else (bw[1] - bw[0]) / (band[1] - band[0])
            rows.append(dict(name=name, rec=st, group=grp, center_ghz=round(c / 1e9, 5), side=side, window=k, visible=round(vis, 3), n_drone_box=len(lines),
                             path=str((OUT / "images" / f"{name}.png").resolve())))
    print(st, len(jobs), flush=True)
D = pd.DataFrame(rows); D.to_csv(ROOT / "runs/review/drff_v8_test.csv", index=False)
(SP / "drff8_all.txt").write_text("\n".join(D.path) + "\n")
for g in ["v100", "v90", "v80", "v70", "v60", "v50", "v30", "nodrone"]:
    (SP / f"drff8_{g}.txt").write_text("\n".join(D[D.group == g].path) + "\n")
(SP / "drff8_main.txt").write_text("\n".join(D[D.group != "v30"].path) + "\n")        # mAP 용: 학습 기준(50 %) 이상 + 드론 없는 그림
print(D.groupby("group").agg(n=("name", "size"), with_box=("n_drone_box", lambda s: int((s > 0).sum())), vis=("visible", "median")))
print("DRFF8_DONE", len(D))
