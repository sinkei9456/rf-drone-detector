"""테스트 ④ — Neo 2 실제 그림에서 drone 검출 (2026-10-09 밤, doc/report_261010.md 2장). 학습·검증·바탕에 쓰지 않은 녹음만:
  ④-1 실내: 실내 Neo 2 ON 250장(모두 "보임") + OFF 250장(드론 없음 → 오경보)
  ④-2 공원: 13(자유비행), 07(30 m), 04·05·06·08·11·12(50 m 안팎). 그림마다 자동 레이블(runs/neo2_labels/labels.csv)의 "보임"/"안 보임".
그림 한 장 기준, 신뢰도 0.25 이상 drone 박스가 있으면 "그림". 위치까지 맞힘 = 정답 박스(드론 채널 × 신호 시간 범위)와 IoU 0.3 이상.
진단용으로 그림마다 가장 높은 drone 신뢰도(0.01 까지)도 저장. 모델 하나씩 불러 평가하고 지운다(GPU 메모리).
출력: runs/eval_neo2/<이름>_pictures.csv, runs/eval_neo2/summary.csv
사용: python src/eval_neo2_test.py --models m1=model/model_01.pt m3a=model/model_03a.pt …"""
import argparse, gc
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "runs/eval_neo2"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); A = ap.parse_args()
import torch
from ultralytics import YOLO
TEST = {"indoor_ON", "indoor_OFF", "13", "07", "04", "05", "06", "08", "11", "12"}
L = pd.read_csv(ROOT / "runs/neo2_labels/labels.csv"); L["key"] = np.where(L.session.str.startswith("indoor"), L.session, L.session.str[:2])
T = L[L.key.isin(TEST)].reset_index(drop=True)


def iou(a, b):
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


summ = []
for spec in A.models:
    name, w = spec.split("=", 1)
    if not (ROOT / w).exists(): print("없음", w); continue
    m = YOLO(str(ROOT / w)); rows = []; paths = [str(ROOT / p) for p in T.path]
    for i in range(0, len(paths), 32):
        for k, r in enumerate(m.predict(paths[i:i + 32], conf=0.01, imgsz=640, verbose=False, batch=32), start=i):
            c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy(); xy = r.boxes.xyxyn.cpu().numpy()
            d = (c == 0); dd = d & (s >= 0.25); t = T.iloc[k]
            gt = (t.x0, 0.25, t.x1, 0.75) if t.state == "visible" else None
            rows.append(dict(key=t.key, state=t.state, image=t.image, max_conf=float(s[d].max()) if d.any() else 0.0, det=bool(dd.any()),
                             hit=bool(gt is not None and any(iou(tuple(b), gt) >= 0.3 for b in xy[dd])), n_box=int(dd.sum())))
    P = pd.DataFrame(rows); P.to_csv(OUT / f"{name}_pictures.csv", index=False)
    for (key, state), q in P.groupby(["key", "state"]):
        summ.append(dict(model=name, key=key, state=state, n=len(q), det=int(q.det.sum()), det_rate=round(q.det.mean(), 4), hit=int(q.hit.sum()),
                         hit_rate=round(q.hit.mean(), 4) if state == "visible" else np.nan))
    print(name, "끝", flush=True); del m; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
S = pd.DataFrame(summ); old = OUT / "summary.csv"
if old.exists():
    O = pd.read_csv(old); S = pd.concat([O[~O.model.isin(S.model)], S])
S.to_csv(old, index=False); print(S.to_string()); print("EVAL_NEO2_DONE", flush=True)
