"""rev1 정답 확인용 이미지 (2026-11-13): 드론 정답 박스를 빨간 사각형으로 그린 이미지.
data/derived/rev1/ 아래 폴더마다 images/·labels/ 옆에 gt_vis/ 를 만든다(gt_vis = ground truth visualization, 정답을 그린 그림).
  gt_vis/<이름>.png = images/<이름>.png 위에 labels/<이름>.txt 의 드론 박스(라벨 번호 0)를 빨간 사각형(선 두께 3픽셀)으로 그린 것.
  와이파이·블루투스 박스(라벨 번호 1·2)는 그리지 않는다. 드론이 없는 이미지는 원래 이미지와 같다.
  혼합 조건 폴더(<fold>_cmix_s<seed>)는 images/ 가 링크이므로 gt_vis/ 도 원래 폴더의 gt_vis/ 를 가리키는 상대 링크로 둔다.
순서: 테스트셋(test2_*, test1_*, test_nodrone_*) → 학습셋(<fold>_c02·c20·c50·c80_s<seed>) → 혼합 조건 링크. 이미 있는 파일은 건너뛴다.
학습이 도는 중에도 돌릴 수 있게 CPU 우선순위를 가장 낮게(nice 19) 둔다(디스크 우선순위는 띄울 때 ionice 로).
사용: python paper/src/rev1_gt_vis.py [--workers 6]   (끝나면 REV1_GT_VIS_DONE)"""
import os, sys, argparse, time
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C
from PIL import Image, ImageDraw


def draw(a):
    img, out = a
    if out.exists(): return 0
    im = Image.open(img).convert("RGB"); d = ImageDraw.Draw(im); W, H = im.size
    lab = Path(str(img).replace("/images/", "/labels/")).with_suffix(".txt")
    for l in open(lab):
        p = l.split()
        if p and p[0] == "0":
            cx, cy, w, h = float(p[1]) * W, float(p[2]) * H, float(p[3]) * W, float(p[4]) * H
            d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(255, 0, 0), width=3)
    tmp = out.with_suffix(".tmp.png"); im.save(tmp); tmp.rename(out); return 1


def order():
    ds = [d for d in sorted(C.DATA.iterdir()) if (d / "images").is_dir() and (d / "labels").is_dir()]
    test = [d for d in ds if d.name.startswith(("test1_2_", "test1_1_", "test_nodrone_"))]
    mix = [d for d in ds if "_cmix_" in d.name]
    return test, [d for d in ds if d not in test and d not in mix], mix


def main(workers):
    os.nice(19); test, train, mix = order(); t0 = time.time()
    with Pool(workers, initializer=os.nice, initargs=(0,)) as P:
        for d in test + train:
            (d / "gt_vis").mkdir(exist_ok=True); imgs = sorted((d / "images").glob("*.png"))
            n = sum(P.imap_unordered(draw, [(i, d / "gt_vis" / i.name) for i in imgs], chunksize=32))
            print(f"{d.name}: 이미지 {len(imgs)}장, 새로 그림 {n}장 ({time.time() - t0:.0f}초)", flush=True)
    for d in mix:
        (d / "gt_vis").mkdir(exist_ok=True); n = 0
        for i in sorted((d / "images").glob("*.png")):
            src = i.resolve().parent.parent / "gt_vis" / i.name; dst = d / "gt_vis" / i.name
            if dst.is_symlink() or dst.exists(): continue
            os.symlink(os.path.relpath(src, dst.parent), dst); n += 1
        print(f"{d.name}: 링크 {n}개", flush=True)
    bad = [d.name for d in test + train + mix if len(list((d / "gt_vis").glob("*.png"))) != len(list((d / "images").glob("*.png")))]
    print("장수가 맞지 않는 폴더:", bad if bad else "없음", flush=True)
    if not bad: print("REV1_GT_VIS_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); main(ap.parse_args().workers)
