"""생성 간섭 모델의 합성 데이터 만들기 (2026-11-12). 원고 3–4장, 미학습 계열 평가. 학습은 하지 않는다(CPU).
이미지 한 장(v10 규칙, src/synth_iq_build._img_job_v9 와 같은 확률):
  15 % 는 신호를 넣지 않고, 나머지의 70 % 에 드론 신호 세그먼트(그중 15 % 는 2개). 드론은 대역폭의 50–100 % 가 보이게(30 % 는 전체), SNR 6–25 dB.
  배경 잡음 = 공원 배경(train 0–1 s, validation 1–2 s, 테스트 2–3 s)에서 무작위 20 ms, 노이즈 전력 1.
  와이파이·블루투스 버스트 개수 = 이미지마다 v10 규칙으로 뽑은 기준 개수(와이파이 N_WIFI, 블루투스 0–30) × 간섭 밀도 배율 k (확률적 반올림).
간섭 밀도 4조건을 같은 이미지로 맞추는 방법(원고 3.3절 "드론 신호 세그먼트, 위치, SNR, 배경 잡음이 같고 간섭 신호의 양만 다르다"):
  드론·배경·기준 개수는 난수 A, 와이파이는 난수 B, 블루투스는 난수 C 로 따로 뽑는다. 버스트는 하나씩 차례로 뽑히므로 개수가 늘어도 앞의 버스트는 같다
  → 밀도가 높은 조건은 낮은 조건의 버스트를 모두 포함한다(그림 5 와 같은 방식).
스펙트럼 점유율(원고 3.3절): 97 × 512 칸, 잡음 세기 = 그 이미지의 배경 잡음(드론·버스트 넣기 전)에서 줄마다 하위 10 %, +6 dB, 드론 박스와 DC 제외.
명령:
  calib  : 간섭 밀도마다 k 를 찾는다. fold 1 train 1,500장(seed 1)의 점유율 중앙값이 목표(2·20·50·80 %)에 맞도록 이분 탐색 → paper/runs/rev1/calib.json
  build  : 학습·검증 데이터 40벌(5 fold × 4 조건 × seed 2) + 미학습 계열 평가(fold × 테스트 밀도 4, 드론 있는 이미지 500) + 드론 없는 이미지(테스트 밀도 4 × 500)
           → data/derived/rev1/<이름>/{images,labels}, meta.json(이미지마다 점유율·버스트 수·드론 출처), 학습 목록·yaml → paper/runs/rev1/{lists,yaml}
  summary: 만든 데이터셋마다 점유율 중앙값 표 → paper/runs/rev1/synth_summary.md
사용: python paper/src/rev1_synth.py calib | build [--only 이름 …] | summary   (끝나면 REV1_CALIB_DONE / REV1_BUILD_DONE / REV1_SUMMARY_DONE)"""
import sys, json, argparse, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C
sys.path.insert(0, str(C.ROOT / "src"))
import synth_iq as S
import occupancy as E
from PIL import Image

_W = {}


def _init():
    if "arr" not in _W:
        _W["arr"], _W["meta"] = C.pieces()
        for k, name in C.BG.items():
            a = np.load(C.ROOT / f"data/derived/hackrf_clean/{name}.npy", mmap_mode="r"); v = np.asarray(a)
            _W[k] = (a, float(np.sqrt(np.mean(np.abs(v - v.mean()) ** 2))))


def compose(role, pool, idx, rep, k, fold_i=0, drone=None):
    """role: train | val | test_drone | test_nodrone. drone=None 이면 v10 확률, True 면 드론 1개(15 % 는 2개) 꼭 넣음, False 면 드론 없음(빈 이미지 없음).
    반환 (iq, 박스 목록, 정보)."""
    _init(); arr, meta = _W["arr"], _W["meta"]
    sa, sw, sb = C.SEED[role]; key = [rep, fold_i, idx] if role in ("train", "val") else ([fold_i, idx] if role == "test_drone" else [idx])
    if role in ("train", "val"): key = [rep, idx]                            # 학습·검증 데이터: fold 와 상관없이 같은 씨앗(드론 출처만 fold 의 모음에서 고름)
    rng = np.random.default_rng([sa] + key)
    if drone is None:
        empty = rng.random() < 0.15; nd = (2 if rng.random() < 0.15 else 1) if (not empty and rng.random() < 0.7) else 0
    else:
        empty = False; nd = (2 if rng.random() < 0.15 else 1) if drone else 0
    a, sd = _W["train" if role == "train" else "val" if role == "val" else "test"]
    off = int(rng.integers(0, len(a) - S.N + 1)); bg = (np.asarray(a[off: off + S.N]) / sd).astype(np.complex64)
    w0, b0 = int(rng.choice(C.N_WIFI)), int(rng.integers(0, C.N_BT_MAX + 1)); uw, ub = rng.random(2)
    x, dboxes, used, taken = bg.copy(), [], [], set()
    for _ in range(nd):
        j = int(rng.choice([q for q in pool if q not in taken])); taken.add(j)
        p = dict(meta[j]); p["iq"] = np.array(arr[j]); snr_t = float(rng.uniform(6, 25))
        fo = S.place_drone_v8(rng, p["bw"]); p["iq"] = S.cut_to_window(p["iq"], fo)
        x, b, snr = S.add_drone(x, p, rng, snr_t, fo=fo); dboxes.append(b)
        used.append(dict(piece=j, group=meta[j]["group"], snr=round(snr, 1), visible_mhz=round((b[4] - b[3]) / 1e6, 2), bw_mhz=round(p["bw"] / 1e6, 2)))
    n_w, n_b = (0, 0) if empty else (int(np.floor(k * w0 + uw)), int(np.floor(k * b0 + ub)))
    x, wb, _ = S.make_background_v8(np.random.default_rng([sw] + key), n_w, 0, base=x)
    x, bb, _ = S.make_background_v8(np.random.default_rng([sb] + key), 0, n_b, base=x)
    occ = occupancy(x, bg, dboxes)
    return x, wb + bb + dboxes, dict(empty=bool(empty), n_drone=nd, n_wifi=n_w, n_bt=n_b, occ=None if occ is None else round(occ, 5), bg=C.BG["train" if role == "train" else "val" if role == "val" else "test"],
                                      bg_offset=off, seed=[sa, sw, sb] + key, pieces=used)


def occupancy(x, bg, dboxes):
    P = E.power(x); floor = np.percentile(E.power(bg), 10, axis=0)
    on = P > floor[None, :] * 10 ** (E.THR_DB / 10); keep = np.broadcast_to(~E.DC[None, :], P.shape).copy()
    for _, t0, t1, lo, hi in dboxes:
        r = (E.FR + E.FS / E.NF / 2 > lo) & (E.FR - E.FS / E.NF / 2 < hi)
        keep[int(np.floor(t0 / S.WIN * 97)):int(np.ceil(t1 / S.WIN * 97)), r] = False
    return float(on[keep].mean()) if keep.mean() >= 0.05 else None             # 드론 경계 상자가 이미지를 거의 다 덮으면(나머지 영역 5 % 미만) 잴 수 없음 → 중앙값에서 뺌


def write(out, stem, x, boxes):
    Image.fromarray(S.to_gray_v7(x)).save(out / "images" / f"{stem}.png"); cls = {"drone": 0, "wifi": 1, "bt": 2}; lines = []
    for b in boxes:
        x0, y0, x1, y1 = S.box_px(b); x0, x1 = max(0, x0) / 640, min(640, x1) / 640; y0, y1 = max(0, y0) / 640, min(640, y1) / 640
        if x1 - x0 < 0.003 or y1 - y0 < 0.003: continue
        lines.append(f"{cls[b[0]]} {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f}")
    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


# ---------------- calib ----------------
def _occ_job(a):
    i, k, pool = a; return compose("train", pool, i, 1, k)[2]["occ"]


def calib(n=1500, workers=16):
    pool = C.split_pieces(C.FOLDS[0][0])["train"]; res = {}
    with Pool(workers, initializer=_init) as P:
        med = lambda k: float(np.median([o for o in P.map(_occ_job, [(i, k, pool) for i in range(n)], chunksize=8) if o is not None]))
        n_na = sum(o is None for o in P.map(_occ_job, [(i, 1.0, pool) for i in range(n)], chunksize=8)); print(f"잴 수 없는 이미지(드론 경계 상자가 이미지를 거의 다 덮음) {n_na}/{n}장", flush=True)
        for cond, tg, name in C.CONDS:
            lo, hi = 0.0, 2.0; best = (9.0, None, None)                         # (|중앙값 − 목표|, k, 중앙값) 중 목표에 가장 가까운 것
            while med(hi) < tg: lo, hi = hi, hi * 2; assert hi < 200, "목표에 닿지 않음"
            for _ in range(16):
                mid = (lo + hi) / 2; m = med(mid); best = min(best, (abs(m - tg), mid, m))
                if m < tg: lo = mid
                else: hi = mid
                if abs(m - tg) < 0.001 + 0.005 * tg: break
            _, k, m = best; res[cond] = dict(target=tg, k=round(k, 4), median=round(m, 4), n=n)
            print(f"{name}({cond}): 목표 {tg * 100:.0f} % → k = {k:.3f}, 중앙값 {m * 100:.2f} %", flush=True)
    C.RUNS.mkdir(parents=True, exist_ok=True); json.dump(res, open(C.RUNS / "calib.json", "w"), ensure_ascii=False, indent=1)
    print("REV1_CALIB_DONE", flush=True)


# ---------------- build ----------------
def _img_job(a):
    out, stem, role, pool, idx, rep, k, fold_i, drone = a
    x, boxes, info = compose(role, pool, idx, rep, k, fold_i, drone); write(Path(out), stem, x, boxes); info["stem"] = stem; return info


def jobs():
    """(이름, [(작업 인자)…]) 목록."""
    K = {c: v["k"] for c, v in json.load(open(C.RUNS / "calib.json")).items()}; meta = C.pieces()[1]; J = []
    for fi, (fk, _, _) in enumerate(C.FOLDS):
        sp = C.split_pieces(fk, meta)
        for cond, _, _ in C.CONDS:
            for rep in C.REPS:
                name = C.set_name(fk, cond, rep); out = str(C.DATA / name)
                J.append((name, [(out, f"train_{i:05d}", "train", sp["train"], i, rep, K[cond], fi, None) for i in range(C.N_TRAIN)]
                          + [(out, f"val_{i:05d}", "val", sp["val"], i, rep, K[cond], fi, None) for i in range(C.N_VAL)]))
            name = C.test_name(fk, cond); out = str(C.DATA / name)
            J.append((name, [(out, f"test_{i:05d}", "test_drone", sp["test"], i, 0, K[cond], fi, True) for i in range(C.N_TEST_DRONE)]))
    for cond, _, _ in C.CONDS:
        name = C.nodrone_name(cond); out = str(C.DATA / name)
        J.append((name, [(out, f"test_{i:05d}", "test_nodrone", [], i, 0, K[cond], 0, False) for i in range(C.N_TEST_NODRONE)]))
    return J


def build(only=None, workers=16):
    (C.RUNS / "lists").mkdir(parents=True, exist_ok=True); (C.RUNS / "yaml").mkdir(parents=True, exist_ok=True)
    with Pool(workers, initializer=_init) as P:
        for name, js in jobs():
            if only and name not in only: continue
            d = C.DATA / name
            if (d / "meta.json").exists(): print("있음, 건너뜀:", name, flush=True); continue
            (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True); t = time.time()
            info = P.map(_img_job, js, chunksize=16)
            json.dump(info, open(d / "meta.json.tmp", "w")); (d / "meta.json.tmp").rename(d / "meta.json")
            occ = np.array([r["occ"] for r in info if r["occ"] is not None]); print(f"{name}: {len(info)}장, 점유율 중앙값 {np.median(occ) * 100:.1f} % (잴 수 없는 이미지 {len(info) - len(occ)}장), {time.time() - t:.0f}초", flush=True)
            if name.startswith("f"):                                               # 학습 목록과 yaml
                for part in ("train", "val"):
                    (C.RUNS / "lists" / f"{name}_{part}.txt").write_text("".join(f"{d / 'images' / (r['stem'] + '.png')}\n" for r in info if r["stem"].startswith(part)))
                (C.RUNS / "yaml" / f"{name}.yaml").write_text(f"train: ../lists/{name}_train.txt\nval: ../lists/{name}_val.txt\nnames:\n  0: drone\n  1: wifi\n  2: bt\n")
    print("REV1_BUILD_DONE", flush=True)


def summary():
    L = ["# 생성 간섭 모델의 합성 데이터 요약", "", "paper/src/rev1_synth.py summary 가 만든 표. 스펙트럼 점유율은 원고 3.3절의 정의(드론 경계 상자·DC 제외).", "",
         "**표 1. 간섭 밀도 배율 (calib.json, fold 1 train 1,500장으로 맞춤)**", "", "| 조건 | 목표 (%) | 배율 k | 맞춘 중앙값 (%) |", "|---|---:|---:|---:|"]
    cal = json.load(open(C.RUNS / "calib.json"))
    for cond, tg, nm in C.CONDS: L.append(f"| {nm} ({cond}) | {tg * 100:.0f} | {cal[cond]['k']:.3f} | {cal[cond]['median'] * 100:.2f} |")
    L += ["", "**표 2. 데이터셋별 이미지 수(장)와 스펙트럼 점유율 중앙값(%)** (잴 수 없는 이미지 = 드론 경계 상자가 이미지를 거의 다 덮어 나머지 영역이 5 % 미만, 중앙값에서 뺌)", "",
          "| 데이터셋 | 이미지 수 | 잴 수 없는 이미지 수 | 점유율 중앙값 | 드론 있는 이미지 비율 (%) | 와이파이 버스트 수 중앙값 | 블루투스 버스트 수 중앙값 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, _ in jobs():
        f = C.DATA / name / "meta.json"
        if not f.exists(): L.append(f"| {name} | (없음) | | | | | |"); continue
        info = json.load(open(f))
        for part in (("train", "val") if name.startswith("f") else ("test",)):
            q = [r for r in info if r["stem"].startswith(part)]
            o = [r['occ'] for r in q if r['occ'] is not None]
            L.append(f"| {name} {part} | {len(q)} | {len(q) - len(o)} | {np.median(o) * 100:.1f} | {np.mean([r['n_drone'] > 0 for r in q]) * 100:.1f} | "
                     f"{np.median([r['n_wifi'] for r in q]):.0f} | {np.median([r['n_bt'] for r in q]):.0f} |")
    (C.RUNS / "synth_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L)); print("REV1_SUMMARY_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["calib", "build", "summary"]); ap.add_argument("--only", nargs="*"); ap.add_argument("--workers", type=int, default=16)
    A = ap.parse_args()
    if A.cmd == "calib": calib(workers=A.workers)
    elif A.cmd == "build": build(A.only, A.workers)
    else: summary()
