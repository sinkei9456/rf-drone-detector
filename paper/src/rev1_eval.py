"""rev1 평가: 모델 하나 (2026-11-12). 원고 3.5절(threshold), 4.1–4.2절(테스트 1-1, baseline, recall·FPR). 테스트 1-2·1-4 은 데이터가 생기면 추가한다.
판정: 20 ms 이미지 한 장마다, threshold 이상인 드론 박스가 하나라도 있으면 양성.
  합성(validation·테스트 1-1): 정답 드론 박스와 IoU 0.3 이상인 드론 박스가 threshold 이상이면 정탐.
  실측(baseline): 드론 박스의 세로 범위(주파수)가 Neo 2 채널(2444.5–2454.5 MHz = 이미지 세로 0.25–0.75)과 IoU 0.3 이상이면 정탐.
이미지마다 저장: max_conf(드론 박스 최고 confidence), max_hit(정답과 맞는 드론 박스 최고 confidence). confidence 0.01 이상 박스만 본다.
  → recall(t) = 드론 있는 이미지 중 max_hit ≥ t 의 비율, FPR(t) = 드론 없는 이미지 중 max_conf ≥ t 의 비율.
threshold: 자기 validation 600장에서 이미지 단위 F1(TP = 드론 이미지 정탐, FN = 드론 이미지 미탐, FP = 드론 없는 이미지 양성)이 가장 큰 값.
  후보 0.01–0.95(0.01 간격), 같은 F1 이면 낮은 값. 테스트 데이터는 보지 않는다. 가중치는 마지막 epoch(last.pt).
baseline(부록 E 표 E-1): recall = 공원 01·03·09·10·13·07(세션별), 실내 ON. 참고 = 공원 04·05·06·08·11·12(약 50 m). FPR = 공원 02 의 2–3 s(f100–f148),
  실내 OFF(10-08), 실내 드론 없음(10-07, 안테나 연결, AntennaOn_NoDrone 01–05 의 20 ms 이미지 250장, 학습과 같은 변환 S.to_gray_v7).
출력: paper/runs/rev1/eval/<이름>/{val.csv, test1_1.csv, test1_3.csv, result.json}
사용: python paper/src/rev1_eval.py --name f1_mini_c02_s1 --weights paper/runs/rev1/models/f1_mini_c02_s1.pt   (끝나면 REV1_EVAL_DONE <이름>)"""
import argparse, json, sys, gc, os
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C
sys.path.insert(0, str(C.ROOT / "src"))
CH = (0.25, 0.75); IOU = 0.3; TS = np.round(np.arange(0.01, 0.951, 0.01), 2)
DEV = os.environ.get("REV1_DEVICE")                                                 # 코드 시험용(예: cpu). 비우면 GPU


def iou2(a, b):
    iw, ih = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1])); i = iw * ih
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def iou1(a0, a1, b0, b1):
    i = max(0.0, min(a1, b1) - max(a0, b0)); return i / (max(a1, b1) - min(a0, b0) + 1e-9)


def gt_boxes(img):
    f = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt"); out = []
    for l in open(f):
        p = l.split()
        if p and p[0] == "0":
            cx, cy, w, h = map(float, p[1:]); out.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return out


def predict(m, src, batch=32):
    """src: 이미지 경로 목록 또는 흑백 배열 목록. 반환: 이미지마다 [(x0, y0, x1, y1, conf)] 드론 박스(정규화 좌표)."""
    out = []
    for i in range(0, len(src), batch):
        b = src[i:i + batch]
        if not isinstance(b[0], str): b = [np.stack([g] * 3, -1) for g in b]
        for r in m.predict(b, conf=0.01, imgsz=640, verbose=False, batch=batch, **({"device": DEV} if DEV else {})):
            c = r.boxes.cls.cpu().numpy().astype(int); s = r.boxes.conf.cpu().numpy(); xy = r.boxes.xyxyn.cpu().numpy()
            out.append([(*map(float, xy[k]), float(s[k])) for k in range(len(c)) if c[k] == 0])
    return out


def syn_rows(m, imgs, **tag):
    rows = []
    for img, boxes in zip(imgs, predict(m, imgs)):
        gt = gt_boxes(img); hit = [b[4] for b in boxes if any(iou2(b[:4], g) >= IOU for g in gt)]
        rows.append(dict(image=Path(img).name, has_drone=bool(gt), max_conf=max([b[4] for b in boxes], default=0.0), max_hit=max(hit, default=0.0), **tag))
    return rows


def rates(d, t):
    p, n = d[d.has_drone], d[~d.has_drone]
    return dict(recall=float((p.max_hit >= t).mean()) if len(p) else None, fpr=float((n.max_conf >= t).mean()) if len(n) else None, n_pos=len(p), n_neg=len(n))


def tune(v):
    p, n = v[v.has_drone], v[~v.has_drone]; best = None
    for t in TS:
        tp = int((p.max_hit >= t).sum()); fn = len(p) - tp; fp = int((n.max_conf >= t).sum()); f1 = 2 * tp / max(2 * tp + fn + fp, 1)
        if best is None or f1 > best[1] + 1e-12: best = (float(t), f1, tp, fn, fp)
    return dict(threshold=best[0], f1=round(best[1], 4), tp=best[2], fn=best[3], fp=best[4], n_pos=len(p), n_neg=len(n))


def baseline(m):
    import synth_iq as S, hackrf_paths as HP
    Lb = pd.read_csv(C.ROOT / "runs/neo2_labels/labels.csv"); Lb["key"] = np.where(Lb.session.str.startswith("indoor"), Lb.session, Lb.session.str[:2])
    Lb["frame"] = Lb.image.str.extract(r"_f(\d+)\.png$")[0].astype(int)
    rec = ["01", "03", "09", "10", "13", "07", "indoor_ON"]; ref = ["04", "05", "06", "08", "11", "12"]
    T = Lb[Lb.key.isin(rec + ref + ["indoor_OFF"]) | ((Lb.key == "02") & (Lb.frame >= 100))].reset_index(drop=True)
    T["role"] = np.where(T.key.isin(rec), "recall", np.where(T.key.isin(ref), "reference", "fpr"))
    rows = []
    for (_, t), boxes in zip(T.iterrows(), predict(m, [str(C.ROOT / p) for p in T.path])):
        hit = [b[4] for b in boxes if iou1(b[1], b[3], *CH) >= IOU]
        rows.append(dict(session=t.key, role=t.role, state=t.state, image=t.image, max_conf=max([b[4] for b in boxes], default=0.0), max_hit=max(hit, default=0.0)))
    for fid in ("01", "02", "03", "04", "05"):                                        # 실내 드론 없음 10-07
        f = [str(p) for p in HP.nodrone_files("AntennaOn_NoDrone", fid)]; assert len(f) == 1, fid
        r = np.fromfile(f[0], dtype=np.int8).astype(np.float32); x = (r[0::2] + 1j * r[1::2]).astype(np.complex64)
        g = [S.to_gray_v7(x[i * S.N:(i + 1) * S.N]) for i in range(len(x) // S.N)]
        for i, boxes in enumerate(predict(m, g)):
            rows.append(dict(session="indoor_nodrone_1007", role="fpr", state="none", image=f"AntennaOn_NoDrone_{fid}_f{i:03d}", max_conf=max([b[4] for b in boxes], default=0.0), max_hit=0.0))
    return pd.DataFrame(rows)


def main(name, weights):
    import torch
    from ultralytics import YOLO
    out = C.RUNS / "eval" / name; out.mkdir(parents=True, exist_ok=True); fk = name.rsplit("_c", 1)[0]; m = YOLO(str(weights))
    val = [str(C.DATA / name / "images" / f"val_{i:05d}.png") for i in range(C.N_VAL)]
    V = pd.DataFrame(syn_rows(m, val, set="val")); V.to_csv(out / "val.csv", index=False); th = tune(V); t = th["threshold"]
    rows = []
    for cond, _, _ in C.CONDS:
        rows += syn_rows(m, [str(C.DATA / C.test_name(fk, cond) / "images" / f"test_{i:05d}.png") for i in range(C.N_TEST_DRONE)], test_density=cond)
        rows += syn_rows(m, [str(C.DATA / C.nodrone_name(cond) / "images" / f"test_{i:05d}.png") for i in range(C.N_TEST_NODRONE)], test_density=cond)
    T1 = pd.DataFrame(rows); T1.to_csv(out / "test1_1.csv", index=False)
    B = baseline(m); B.to_csv(out / "test1_3.csv", index=False)
    res = dict(name=name, weights=str(Path(weights).relative_to(C.ROOT)) if Path(weights).is_absolute() else str(weights), val=th,
               test1={c: rates(T1[T1.test_density == c], t) for c, _, _ in C.CONDS},
               baseline_recall={s: dict(all=float((q.max_hit >= t).mean()), visible=float((q[q.state == "visible"].max_hit >= t).mean()) if (q.state == "visible").any() else None,
                                        n=len(q), n_visible=int((q.state == "visible").sum())) for s, q in B[B.role != "fpr"].groupby("session")},
               baseline_fpr={s: dict(fpr=float((q.max_conf >= t).mean()), n=len(q)) for s, q in B[B.role == "fpr"].groupby("session")})
    json.dump(res, open(out / "result.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(dict(threshold=t, val_f1=th["f1"], test1={c: (round(v["recall"], 3), round(v["fpr"], 3)) for c, v in res["test1"].items()}), ensure_ascii=False))
    del m; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("REV1_EVAL_DONE", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--weights", required=True); A = ap.parse_args()
    main(A.name, A.weights)
