"""통합 보고서(report_261006-08)용 그림 한 장 기준 평가 (2026-10-07 밤). 모델 하나 = 프로세스 하나. last.pt, 신뢰도 0.25, drone 박스만 센다.
테스트셋(평가에만, 세 모델 같음):
  ① HackRF 실측: 안테나 뺀 넷째·다섯째 녹음 100장 + 안테나 끼운 녹음 5개 250장(드론 없음) → 오경보율. 그림마다 박스를 <name>_hackrf_boxes.csv 로 저장
  ② DRFF-R2 20 MHz 실제 그림 1,840장(src/drff_v8_test.py) → 보이는 비율별 검출률·위치까지 맞힌 검출률(IoU 0.3 이상), 드론 없는 그림 오경보율,
     drone mAP50(30 % 뺀 1,640장). 그림마다 결과를 <name>_drff_pictures.csv 로 저장
  ③ 합성 테스트 1,000장(세 종류 라벨, split_v6/syn8_test.txt) → 검출률·위치·오경보율·보이는 비율별, mAP50(drone 하나만 찾는 모델은 drone 만)
계산은 src/eval_v13.py·eval_v14.py 와 같다. 출력: runs/eval_v15/<name>.json
사용: python src/eval_v15.py --name m2 --weights runs/det_v15_b/train/weights/last.pt [--drone_only]"""
import argparse, json, gc, glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import sys as _sys; _sys.path.insert(0, str(ROOT / "src")); import hackrf_paths as HP   # 10-09 HackRF 폴더 정리: 녹음 경로는 hackrf_paths 에서
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--weights", required=True); ap.add_argument("--drone_only", action="store_true"); A = ap.parse_args()
import torch
from ultralytics import YOLO
import synth_iq as S
OUTD = ROOT / "runs/eval_v15"; OUTD.mkdir(exist_ok=True); D = ROOT / "data/derived/split_v6"; CONF = 0.25


def gt(img):
    out = []
    for l in open(Path(img.replace("/images/", "/labels/")).with_suffix(".txt")):
        w = l.split()
        if len(w) == 5 and w[0] == "0": xc, yc, bw, bh = map(float, w[1:]); out.append(((xc - bw / 2) * 640, (yc - bh / 2) * 640, (xc + bw / 2) * 640, (yc + bh / 2) * 640))
    return out


def iou(a, b):
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def predict(m, imgs):
    rows = []
    for i in range(0, len(imgs), 16):
        for p, r in zip(imgs[i:i + 16], m.predict(imgs[i:i + 16], conf=CONF, imgsz=640, verbose=False, batch=16)):
            pr = [b for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()) if int(c) == 0]; g = gt(p)
            rows.append(dict(path=p, img=Path(p).name, has=len(g) > 0, det=len(pr) > 0, hit=any(iou(b, x) >= 0.3 for b in pr for x in g)))
    return pd.DataFrame(rows)


def summ(d):
    pos, neg = d[d.has], d[~d.has]; f = lambda s: round(float(s.mean()), 4) if len(s) else None
    return dict(n=len(d), n_pos=len(pos), n_neg=len(neg), recall=f(pos.det), recall_loc=f(pos.hit), fpr=f(neg.det), fp=int(neg.det.sum()), miss=int((~pos.det).sum()))


def ap50(m, yaml, tag):
    r = m.val(data=str(ROOT / "data/yaml" / yaml), imgsz=640, batch=16, plots=False, verbose=False, project=str(OUTD / "val"), name=f"{A.name}_{tag}", exist_ok=True)
    names = {0: "drone", 1: "wifi", 2: "bt"}; out = {names[int(c)]: round(float(v), 4) for c, v in zip(r.box.ap_class_index, r.box.ap50)}
    del r; gc.collect(); return out


m = YOLO(str(ROOT / A.weights)); res = dict(name=A.name, weights=A.weights, conf=CONF)
# ③ 합성 테스트
s = predict(m, [l.strip() for l in open(D / "syn8_test.txt") if l.strip()])
meta = {r["stem"] + ".png": r for r in json.load(open(ROOT / "data/derived/yolo_syn_iq_v8/meta.json"))}
s["vis"] = [max([u["visible_mhz"] / u["bw_mhz"] for u in meta[i]["pieces"]], default=0) for i in s.img]
s["is_empty"] = [meta[i]["empty"] for i in s.img]
res["syn"] = summ(s); res["syn_empty_fpr"] = summ(s[s.is_empty])["fpr"]
res["syn_by_vis"] = {f"{lo}-{hi}": summ(s[s.has & (s.vis >= lo) & (s.vis < hi)]) for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.01))}
# ② DRFF-R2 20 MHz 실제 그림
T = pd.read_csv(ROOT / "runs/review/drff_v8_test.csv"); d = predict(m, list(T.path)).merge(T[["path", "group", "visible", "rec", "center_ghz"]], on="path")
d.to_csv(OUTD / f"{A.name}_drff_pictures.csv", index=False)
res["drff_by_group"] = {g: summ(d[d.group == g]) for g in ["v100", "v90", "v80", "v70", "v60", "v50", "v30", "nodrone"]}
res["drff_vis50plus"] = summ(d[d.group.isin(["v100", "v90", "v80", "v70", "v60", "v50"]) & d.has]); res["drff_nodrone"] = summ(d[d.group == "nodrone"])
# ① HackRF 실측(드론 없음): 그림 만드는 법은 학습·시연과 같은 S.to_gray_v7
rows = []
for c, fid in [("AntennaOff", "04"), ("AntennaOff", "05")] + [("AntennaOn_NoDrone", f"0{k}") for k in range(1, 6)]:
    f = [str(p) for p in HP.nodrone_files(c, fid)]; assert len(f) == 1, (c, fid)
    r = np.fromfile(f[0], dtype=np.int8).astype(np.float32); x = (r[0::2] + 1j * r[1::2]).astype(np.complex64)
    g = [S.to_gray_v7(x[i * S.N:(i + 1) * S.N]) for i in range(len(x) // S.N)]
    for i, q in enumerate(m.predict([np.stack([u] * 3, -1) for u in g], imgsz=640, conf=CONF, verbose=False, batch=25)):
        bl = list(zip(q.boxes.cls.tolist(), q.boxes.conf.tolist(), q.boxes.xyxy.tolist()))
        for cl, sc, (x0, y0, x1, y1) in bl: rows.append(dict(model=A.name, cond=c, file=fid, frame=i, cls=int(cl), conf=sc, y0=y0, y1=y1, area=(x1 - x0) * (y1 - y0) / 640 ** 2))
        if not bl: rows.append(dict(model=A.name, cond=c, file=fid, frame=i, cls=-1, conf=0, y0=0, y1=0, area=0))
H = pd.DataFrame(rows); H.to_csv(OUTD / f"{A.name}_hackrf_boxes.csv", index=False)
P = H.groupby(["cond", "file", "frame"]).apply(lambda q: bool((q.cls == 0).any()), include_groups=False).reset_index(name="det")
res["hackrf"] = {"off_fp": int(P[P.cond == "AntennaOff"].det.sum()), "off_n": int((P.cond == "AntennaOff").sum()),
                 "on_fp": int(P[P.cond != "AntennaOff"].det.sum()), "on_n": int((P.cond != "AntennaOff").sum()), "fpr": round(float(P.det.mean()), 4)}
pre = "v14" if A.drone_only else "v13"
res["ap50"] = {"syn": ap50(m, f"{pre}_test_syn.yaml", "syn"), "drff": ap50(m, f"{pre}_test_drff.yaml", "drff")}
json.dump(res, open(OUTD / f"{A.name}.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(res, ensure_ascii=False, indent=1))
del m; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("EVAL_DONE", A.name, flush=True)
