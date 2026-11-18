"""20 ms 스펙트로그램 그림(640×640 PNG) → YOLO 모델 → 박스 결과.

입력: 그림 폴더(안의 .png 를 모두 찾음, 보통 src/iq_to_image.py 가 만든 work/images). 폴더 안의 index.csv 가 있으면
      그림마다 중심 주파수·시각을 읽어 박스를 MHz·ms 로 바꿔 적는다(없으면 픽셀 위치만 적음).
모델: 기본 models/model_02a.pt(drone·wifi·bt, Neo 2 를 넣어 다시 학습, 추천 — README 7장). model_03·03a 는 drone 하나만 찾는다.
판정: 그림 한 장마다 신뢰도 0.25 이상인 drone 박스가 하나라도 있으면 "드론 있음". 여러 장을 묶어 판정하는 규칙은 없다.
출력(<out>/):
  detections.csv : 박스 하나에 한 줄 — 그림, 종류, 신뢰도, 픽셀 위치, 시각 범위(ms), 주파수 범위(MHz)
  pictures.csv   : 그림 하나에 한 줄 — drone·wifi·bt 박스 수, 가장 높은 drone 신뢰도, 드론 있음/없음
  summary.txt    : 녹음(원본 파일)마다 "드론 있음" 그림 수 / 전체 그림 수, 걸린 시간
  images/        : 박스를 그린 그림(빨강 drone, 초록 wifi, 파랑 bt). --no_draw 면 만들지 않음
사용(CPU, 라즈베리 파이): python src/predict.py --images work/images --model models/model_02a.pt --device cpu"""
import argparse, csv, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

PX, BW_MHZ, WIN_MS = 640, 20.0, 20.0     # 그림 640 픽셀 = 세로 20 MHz, 가로 20 ms
COL = {"drone": (227, 73, 72), "wifi": (27, 175, 122), "bt": (42, 120, 214)}


def load_index(root):
    """root 아래 index.csv 들을 모두 읽어 {그림 경로(root 기준): 행}."""
    idx = {}
    for f in root.rglob("index.csv"):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            idx[str((f.parent / r["image"]).resolve())] = r
    return idx


def main():
    ap = argparse.ArgumentParser(description="스펙트로그램 그림 → YOLO drone(·wifi·bt) 박스")
    ap.add_argument("--images", default="work/images", help="그림 폴더(안을 모두 찾음) 또는 그림 하나. 기본 work/images")
    ap.add_argument("--model", default="models/model_02a.pt", help="모델 파일. 기본 models/model_02a.pt(추천)")
    ap.add_argument("--out", default=None, help="결과 폴더. 기본 output/<모델 이름>_<날짜_시각>")
    ap.add_argument("--conf", type=float, default=0.25, help="신뢰도 기준. 기본 0.25(보고서와 같음)")
    ap.add_argument("--device", default="cpu", help="cpu(라즈베리 파이) 또는 0(GPU 번호). 기본 cpu")
    ap.add_argument("--batch", type=int, default=1, help="한 번에 넣는 그림 수. 메모리가 작으면 1. 기본 1")
    ap.add_argument("--no_draw", action="store_true", help="박스 그린 그림을 만들지 않음(빠름)")
    a = ap.parse_args()
    from ultralytics import YOLO
    src = Path(a.images); out = Path(a.out or f"output/{Path(a.model).stem}_{datetime.now():%Y%m%d_%H%M%S}")
    imgs = [src] if src.is_file() else sorted(p for p in src.rglob("*.png") if out.resolve() not in p.resolve().parents)   # 결과 폴더 안의 그림은 입력에서 뺌
    if not imgs: sys.exit(f"[멈춤] {src} 에 .png 그림이 없음. 먼저 src/iq_to_image.py 를 실행하세요.")
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_draw: (out / "images").mkdir(exist_ok=True)
    idx = load_index(src if src.is_dir() else src.parent)
    m = YOLO(a.model); names = m.names
    det_rows, pic_rows = [], []; t0 = time.time()
    for b in range(0, len(imgs), a.batch):
        chunk = imgs[b:b + a.batch]
        for p, r in zip(chunk, m.predict([str(q) for q in chunk], imgsz=PX, conf=a.conf, device=a.device, verbose=False, batch=a.batch)):
            rel = p.relative_to(src) if src.is_dir() else Path(p.name); meta = idx.get(str(p.resolve()))
            cnt = {"drone": 0, "wifi": 0, "bt": 0}; best = 0.0; boxes = []
            for c, s, (x0, y0, x1, y1) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                k = names[int(c)]; cnt[k] = cnt.get(k, 0) + 1; boxes.append((k, s, x0, y0, x1, y1))
                if k == "drone": best = max(best, s)
                row = dict(image=str(rel), cls=k, conf=round(s, 4), x0=round(x0, 1), y0=round(y0, 1), x1=round(x1, 1), y1=round(y1, 1), t_start_ms="", t_end_ms="", f_low_mhz="", f_high_mhz="")
                if meta:   # 위 = 중심 + 10 MHz, 아래 = 중심 − 10 MHz, 왼쪽 = 그림 시작 시각
                    fc = float(meta["fc_hz"]) / 1e6; ts = float(meta["t_start_ms"])
                    row.update(t_start_ms=round(ts + x0 / PX * WIN_MS, 2), t_end_ms=round(ts + x1 / PX * WIN_MS, 2),
                               f_low_mhz=round(fc + (0.5 - y1 / PX) * BW_MHZ, 3), f_high_mhz=round(fc + (0.5 - y0 / PX) * BW_MHZ, 3))
                det_rows.append(row)
            pic_rows.append(dict(image=str(rel), source=meta["source"] if meta else "", frame=meta["frame"] if meta else "",
                                 n_drone=cnt.get("drone", 0), n_wifi=cnt.get("wifi", 0), n_bt=cnt.get("bt", 0), max_drone_conf=round(best, 4), drone="yes" if cnt.get("drone", 0) else "no"))
            if not a.no_draw:
                im = Image.open(p).convert("RGB"); dr = ImageDraw.Draw(im)
                for k, s, x0, y0, x1, y1 in boxes:
                    dr.rectangle([x0, y0, x1, y1], outline=COL.get(k, (255, 255, 0)), width=3); dr.text((x0 + 4, y0 + 4), f"{k} {s:.2f}", fill=COL.get(k, (255, 255, 0)))
                q = out / "images" / rel; q.parent.mkdir(parents=True, exist_ok=True); im.save(q)
        if (b // a.batch) % 50 == 0: print(f"  {min(b + a.batch, len(imgs))}/{len(imgs)} 장", flush=True)
    dt = time.time() - t0
    with open(out / "detections.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "cls", "conf", "x0", "y0", "x1", "y1", "t_start_ms", "t_end_ms", "f_low_mhz", "f_high_mhz"]); w.writeheader(); w.writerows(det_rows)
    with open(out / "pictures.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["image", "source", "frame", "n_drone", "n_wifi", "n_bt", "max_drone_conf", "drone"]); w.writeheader(); w.writerows(pic_rows)
    L = [f"모델 {a.model}, 신뢰도 {a.conf}, 장치 {a.device}, 그림 {len(imgs)}장, {dt:.1f} 초(그림 한 장 {1e3 * dt / len(imgs):.0f} ms)", "",
         "원본(녹음 또는 그림 폴더)마다 '드론 있음' 그림 수 / 전체 그림 수:"]
    groups = {}
    for r in pic_rows: groups.setdefault(Path(r["source"]).name if r["source"] else str(Path(r["image"]).parent), []).append(r["drone"] == "yes")
    for g, v in groups.items(): L.append(f"  {g}: {sum(v)} / {len(v)}")
    L.append(f"\n전체: {sum(r['drone'] == 'yes' for r in pic_rows)} / {len(pic_rows)}")
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L)); print(f"결과: {out}/")


if __name__ == "__main__":
    main()
