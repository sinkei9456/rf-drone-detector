"""YOLO 학습 + 실측 평가. 사용: python train_det.py --data ../data/syn.yaml --model yolo26s.pt --out ../runs/det_syn --epochs 60"""
import argparse, json
from pathlib import Path
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--model", default="yolo26s.pt")
ap.add_argument("--out", required=True); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=640); ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--eval", nargs="*", default=[], help="추가 평가 yaml 들"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save_period", type=int, default=-1)
a = ap.parse_args()
out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)   # ultralytics 는 상대경로의 ".." 를 지우므로 절대경로 필수
m = YOLO(str(Path(__file__).parent / a.model))
m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, save_period=a.save_period, project=str(out), name="train", exist_ok=True,
        fliplr=0.0, flipud=0.0, mosaic=0.5, hsv_h=0, hsv_s=0, hsv_v=0.3, scale=0.2, translate=0.1, degrees=0,
        pretrained=True, verbose=False, plots=True)
best = YOLO(str(out / "train" / "weights" / "best.pt"))
res = {}
for y in [a.data] + a.eval:
    r = best.val(data=y, imgsz=a.imgsz, batch=a.batch, project=str(out), name="val_" + Path(y).stem, exist_ok=True, plots=True, verbose=False)
    res[Path(y).stem] = dict(mAP50=float(r.box.map50), mAP5095=float(r.box.map), P=float(r.box.mp), R=float(r.box.mr))
json.dump(res, open(out / "results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
