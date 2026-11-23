"""Neo 2 합성 평가(코드 이름 test1_2): 학습에 쓰지 않은 Neo 2 의 드론 신호 세그먼트를 합성 이미지에 넣은 평가 데이터 만들기와 평가 (2026-11-13).
원고 5.3절, 보충 자료 S3. 짧은 버스트 : 긴 버스트 = 5 : 5.

build: 간섭 밀도 4단계(2·20·50·80 %)마다 드론 있는 이미지 500장 → data/derived/rev1/test2_<밀도>/{images,labels,meta.json}
  - 0–249번 = 짧은 버스트(공원 01, 비행하지 않음, 조종기와 약 1 m), 250–499번 = 긴 버스트(공원 03·09·10, 비행 중).
  - 드론 신호 세그먼트: runs/neo2_pieces.npy(src/neo2_pieces.py 가 만든 20 ms IQ, 드론 채널 ±5 MHz 만 남김)에서
    드론 대역 세기가 노이즈보다 1.5 dB 이상인 것만 쓴다(표 C-2 의 수: 01 299, 03 410, 09 496, 10 498).
  - 세그먼트 처리: 버스트가 켜진 시간만 남기고 꺼진 시간은 0 으로 지운다(경계 20 µs 는 부드럽게 줄임), 그 뒤 다시 ±5 MHz 만 남긴다.
    버스트 찾기는 paper/src/neo2_burst_sessions.py 와 같은 규칙(51 µs 이동 평균, 하위 10 %와 상위 10 %의 가운데를 넘는 시간, 0.1 ms 보다 짧은 틈은 잇고
    0.05 ms 보다 짧은 버스트는 버림). 정답 박스의 시간 범위 = 첫 버스트 시작–마지막 버스트 끝, 주파수 범위 = 드론 채널 10 MHz.
    이유: 긴 버스트 세그먼트는 원래 드론 대역 세기가 노이즈보다 약 2.5 dB 라(보충 자료 그림 S4), 그대로 SNR 6–25 dB 로 키우면 세그먼트에 함께 든
    수신기 잡음이 10 MHz 띠로 같이 커진다. 학습 데이터와 같은 함수(src/synth_iq.add_drone)는 이 노이즈를 배경 잡음의 30 % 아래로 깎느라
    SNR 이 약 −3 dB 에서 멈춘다. 버스트 안의 수신기 잡음은 남는다(한계로 보충 자료 S3 에 적음).
  - 나머지는 미학습 계열 평가(paper/src/rev1_synth.py compose, drone=True)와 같은 규칙과 같은 난수 순서:
    배경 잡음 = 공원 배경 2–3 s(Park02_3)에서 무작위 20 ms, 드론 1개(15 % 는 2개), 위치 src/synth_iq.place_drone_v8, SNR 6–25 dB 무작위,
    와이파이·블루투스 개수 = 기준 개수 × 간섭 밀도 배율 k(paper/runs/rev1/calib.json).
  - 난수 씨앗: 미학습 계열 평가 드론 있는 이미지와 같은 첫 숫자(21·25·26)에 fold 번호(0–4) 대신 9 → 미학습 계열 평가의 어느 이미지와도, 학습·검증 데이터(11–16)과도 겹치지 않음.
  - 드론 없는 이미지는 미학습 계열 평가와 공통(data/derived/rev1/test_nodrone_<밀도>)이라, Neo 2 합성 평가의 FPR 은 미학습 계열 평가의 FPR 과 같다. Neo 2 합성 평가는 recall 만 보고한다.
  - 점검 결과 → paper/runs/rev1/test1_2_build.md
eval: 모델마다 자기 threshold(paper/runs/rev1/eval/<이름>/result.json 의 val.threshold, validation 에서 정한 값)로
  짧은 버스트·긴 버스트의 recall 을 따로 → paper/runs/rev1/eval/<이름>/{test1_2.csv, test1_2.json}. 판정은 rev1_eval.py 와 같다(IoU 0.3 이상인 드론 박스).
사용: python paper/src/rev1_test1_2.py build          (끝나면 REV1_T2_BUILD_DONE)
      python paper/src/rev1_test1_2.py eval [--names 이름 …]   (모델마다 REV1_T2_DONE <이름>, 이미 있으면 건너뜀)"""
import sys, json, argparse, time, os
from pathlib import Path
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C
import rev1_synth as R
sys.path.insert(0, str(C.ROOT / "src"))
import synth_iq as S

FOLD_I = 9                                                    # 난수 씨앗에서 fold 번호 자리(미학습 계열 평가는 0–4)
N_SHORT, N_LONG = 250, 250
SHORT, LONG = ("01",), ("03", "09", "10")
MIN_DB = 1.5                                                  # 드론 대역 세기 하한(표 C-2, 해커톤 합성 규칙 NEO2_MIN_POOL_DB 와 같음)
BW = 10e6
FR = np.fft.fftfreq(S.N, 1 / S.FS)
_N = {}


def t2_name(cond): return f"test1_2_{cond}"


BURSTS = C.DATA / "test1_2_neo2_bursts.json"                   # 세그먼트마다 찾은 버스트 위치(한 번만 계산, build 가 만듦)


def _init():
    R._init()
    if "arr" not in _N:
        meta = json.load(open(C.ROOT / "runs/neo2_pieces_meta.json"))
        _N["arr"], _N["meta"] = np.load(C.ROOT / "runs/neo2_pieces.npy", mmap_mode="r"), meta
        _N["pool0"] = {s: [j for j, m in enumerate(meta) if m["session"][:2] in ss and m["s0_db"] >= MIN_DB] for s, ss in (("short", SHORT), ("long", LONG))}
        _N["cache"] = {}
        if BURSTS.exists():
            _N["bursts"] = {int(j): v for j, v in json.load(open(BURSTS)).items()}
            _N["pool"] = {s: [j for j in v if _N["bursts"][j]] for s, v in _N["pool0"].items()}   # 버스트를 못 찾은 세그먼트는 뺌


def bursts(y):
    """버스트 [(시작, 끝) 샘플]. paper/src/neo2_burst_sessions.py 와 같은 규칙. 켜짐·꺼짐 세기 차이가 1 dB 미만이면 [].
    이동 평균은 np.convolve(…, np.ones(1024) / 1024, "same") 과 같은 값을 누적합으로 계산(빠름)."""
    a = np.abs(y).astype(np.float64) ** 2; c = np.r_[0.0, np.cumsum(a)]; k = np.arange(len(a))
    e = (c[np.minimum(k + 512, len(a))] - c[np.maximum(k - 512, 0)]) / 1024; lo, hi = np.quantile(e, 0.1), np.quantile(e, 0.9)
    if 10 * np.log10(hi / lo) < 1.0: return []
    on = e > (lo + hi) / 2; d = np.diff(np.r_[0, on.astype(int), 0]); out = []
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if out and a - out[-1][1] < 0.1e-3 * S.FS: out[-1][1] = int(b)
        else: out.append([int(a), int(b)])
    return [(a, b) for a, b in out if b - a >= 0.05e-3 * S.FS]


def _burst_job(j):
    _init(); return j, bursts(np.asarray(_N["arr"][j]))


def find_bursts(workers=8):
    """세그먼트마다 버스트 위치를 한 번 찾아 저장 → BURSTS."""
    if BURSTS.exists(): return
    _init(); js = _N["pool0"]["short"] + _N["pool0"]["long"]; t = time.time()
    with Pool(workers) as P: res = dict(P.map(_burst_job, js, chunksize=8))
    BURSTS.parent.mkdir(parents=True, exist_ok=True); json.dump({str(j): v for j, v in res.items()}, open(BURSTS, "w"))
    print(f"버스트 찾기: 세그먼트 {len(js)}개, 못 찾은 것 {sum(not v for v in res.values())}개, {time.time() - t:.0f}초", flush=True)
    _N.clear()


def gated(j, ramp_us=20):
    """세그먼트 j 에서 버스트 시간만 남긴 것. 반환 dict(iq, bw, burst, noise, t0, t1, n_burst, on_frac) 또는 None(버스트를 못 찾음)."""
    if j in _N["cache"]: return _N["cache"][j]
    p = np.asarray(_N["arr"][j]); b = _N["bursts"][j]; out = None
    if b:
        w = np.zeros(S.N, np.float32); r = int(ramp_us * 1e-6 * S.FS); ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))).astype(np.float32)
        for a, e in b:
            w[a:e] = 1
            if a - r >= 0: w[a - r:a] = np.maximum(w[a - r:a], ramp)
            if e + r <= S.N: w[e:e + r] = np.maximum(w[e:e + r], ramp[::-1])
        F = np.fft.fft(p * w); F[np.abs(FR) > BW / 2] = 0; y = np.fft.ifft(F).astype(np.complex64)
        on = np.zeros(S.N, bool)
        for a, e in b: on[a:e] = True
        out = dict(iq=y, bw=BW, burst=float(np.mean(np.abs(y[on]) ** 2)), noise=0.0, t0=b[0][0] / S.FS, t1=b[-1][1] / S.FS, n_burst=len(b), on_frac=float(on.mean()))
    _N["cache"][j] = out; return out


def compose2(idx, k):
    """미학습 계열 평가의 compose(role="test_drone", drone=True)와 같은 난수 순서, 드론 신호 세그먼트만 Neo 2. 반환 (iq, 박스 목록, 정보)."""
    _init(); sa, sw, sb = C.SEED["test_drone"]; key = [FOLD_I, idx]; rng = np.random.default_rng([sa] + key)
    nd = 2 if rng.random() < 0.15 else 1
    a, sd = R._W["test"]; off = int(rng.integers(0, len(a) - S.N + 1)); bg = (np.asarray(a[off: off + S.N]) / sd).astype(np.complex64)
    w0, b0 = int(rng.choice(C.N_WIFI)), int(rng.integers(0, C.N_BT_MAX + 1)); uw, ub = rng.random(2)
    shape = "short" if idx < N_SHORT else "long"; pool = _N["pool"][shape]
    x, dboxes, used, taken = bg.copy(), [], [], set()
    for _ in range(nd):
        j = int(rng.choice([q for q in pool if q not in taken])); taken.add(j); g = gated(j)
        p = dict(g); snr_t = float(rng.uniform(6, 25)); fo = S.place_drone_v8(rng, p["bw"]); p["iq"] = S.cut_to_window(p["iq"], fo)
        x, b, snr = S.add_drone(x, p, rng, snr_t, fo=fo); dboxes.append(b)
        used.append(dict(neo2=j, session=_N["meta"][j]["session"][:2], s0_db=round(_N["meta"][j]["s0_db"], 2), snr=round(snr, 1), snr_target=round(snr_t, 1),
                         n_burst=g["n_burst"], on_frac=round(g["on_frac"], 3), visible_mhz=round((b[4] - b[3]) / 1e6, 2)))
    n_w, n_b = int(np.floor(k * w0 + uw)), int(np.floor(k * b0 + ub))
    x, wb, _ = S.make_background_v8(np.random.default_rng([sw] + key), n_w, 0, base=x)
    x, bb, _ = S.make_background_v8(np.random.default_rng([sb] + key), 0, n_b, base=x)
    occ = R.occupancy(x, bg, dboxes)
    return x, wb + bb + dboxes, dict(shape=shape, n_drone=nd, n_wifi=n_w, n_bt=n_b, occ=None if occ is None else round(occ, 5), bg=C.BG["test"],
                                      bg_offset=off, seed=[sa, sw, sb] + key, pieces=used)


def _job(a):
    out, i, k = a; x, boxes, info = compose2(i, k); R.write(Path(out), f"test_{i:05d}", x, boxes); info["stem"] = f"test_{i:05d}"; return info


def build(workers=8):
    find_bursts(workers); _init(); K = {c: v["k"] for c, v in json.load(open(C.RUNS / "calib.json")).items()}
    n_bad = {s: len(_N["pool0"][s]) - len(_N["pool"][s]) for s in ("short", "long")}
    L = ["# Neo 2 합성 평가 데이터 점검", "", "paper/src/rev1_test1_2.py build 가 만든 표.", "",
         "**표 1. 쓸 수 있는 Neo 2 드론 신호 세그먼트 (단위: 개)**", "", "| 신호 모양 | 공원 세션 | 드론 대역 세기 1.5 dB 이상 | 그중 버스트를 찾지 못해 뺀 것 |", "|---|---|---:|---:|"]
    for s, ss in (("short", SHORT), ("long", LONG)):
        L.append(f"| {'짧은 버스트' if s == 'short' else '긴 버스트'} | {', '.join(ss)} | {len(_N['pool0'][s])} | {n_bad[s]} |")
    L += ["", "**표 2. 간섭 밀도별 Neo 2 합성 평가 이미지 (이미지 수의 단위: 장, 점유율·SNR 의 단위: %·dB)**", "",
          "| 간섭 밀도 | 이미지 수 | 점유율 중앙값 (%) | 짧은 버스트 이미지 | 긴 버스트 이미지 | 드론 2개 이미지 | 실제 SNR 범위 (dB) | 목표와 실제 SNR 차이 최대 (dB) |", "|---|---:|---:|---:|---:|---:|---|---:|"]
    bad = []
    with Pool(workers, initializer=_init) as P:
        for cond, _, cname in C.CONDS:
            d = C.DATA / t2_name(cond)
            if (d / "meta.json").exists(): print("있음, 건너뜀:", d.name, flush=True); info = json.load(open(d / "meta.json"))
            else:
                (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True); t = time.time()
                info = P.map(_job, [(str(d), i, K[cond]) for i in range(N_SHORT + N_LONG)], chunksize=8)
                json.dump(info, open(d / "meta.json.tmp", "w")); (d / "meta.json.tmp").rename(d / "meta.json"); print(f"{d.name}: {len(info)}장, {time.time() - t:.0f}초", flush=True)
            occ = [r["occ"] for r in info if r["occ"] is not None]; ps = [p for r in info for p in r["pieces"]]
            snr = [p["snr"] for p in ps]; gap = max(abs(p["snr"] - p["snr_target"]) for p in ps)
            L.append(f"| {cname} ({cond}) | {len(info)} | {np.median(occ) * 100:.1f} | {sum(r['shape'] == 'short' for r in info)} | {sum(r['shape'] == 'long' for r in info)} | "
                     f"{sum(r['n_drone'] == 2 for r in info)} | {min(snr):.1f}–{max(snr):.1f} | {gap:.2f} |")
            # 점검: 배경 잡음은 테스트용만, 씨앗은 미학습 계열 평가·학습·검증 데이터와 다름, 세그먼트 출처는 정한 세션만
            if {r["bg"] for r in info} != {C.BG["test"]}: bad.append(f"{cond}: 배경 잡음이 {C.BG['test']} 밖")
            if any(r["seed"][3] != FOLD_I or r["seed"][:3] != list(C.SEED["test_drone"]) for r in info): bad.append(f"{cond}: 난수 씨앗이 정한 값과 다름")
            if any((p["session"] in SHORT) != (r["shape"] == "short") for r in info for p in r["pieces"]): bad.append(f"{cond}: 신호 모양과 세션이 맞지 않음")
            if gap > 0.5: bad.append(f"{cond}: 실제 SNR 이 목표와 {gap:.2f} dB 다름")
    trainval = sorted(set(C.SEED["train"]) | set(C.SEED["val"]))
    if set(C.SEED["test_drone"]) & set(trainval): bad.append("난수 씨앗 첫 숫자가 학습·검증 데이터와 겹침")
    L += ["", f"- 난수 씨앗 첫 숫자 {list(C.SEED['test_drone'])}, fold 자리 {FOLD_I}(미학습 계열 평가는 0–4). 학습·검증 데이터의 난수 첫 숫자 {trainval} 와 다름.",
          "- 점유율 중앙값이 목표(2·20·50·80 %)보다 조금 높은 것은 미학습 계열 평가와 같다. train 에는 간섭을 넣지 않은 빈 이미지가 15 % 있어 중앙값을 낮추지만, 평가 이미지에는 빈 이미지가 없다.",
          f"- 배경 잡음은 {C.BG['test']}(공원 배경 2–3 s)만.", "", f"**점검 결과: {'모두 통과' if not bad else '실패 ' + str(len(bad)) + '개'}**"] + [f"- 실패: {b}" for b in bad]
    (C.RUNS / "test1_2_build.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L), flush=True)
    if not bad: print("REV1_T2_BUILD_DONE", flush=True)


def evaluate(names=None):
    import gc, torch
    from ultralytics import YOLO
    import rev1_eval as E
    names = names or sorted(p.parent.name for p in (C.RUNS / "eval").glob("*/result.json"))
    for name in names:
        out = C.RUNS / "eval" / name
        if (out / "test1_2.json").exists(): print("있음, 건너뜀:", name, flush=True); continue
        res = json.load(open(out / "result.json")); t = res["val"]["threshold"]; m = YOLO(str(C.RUNS / "models" / f"{name}.pt")); rows = []
        for cond, _, _ in C.CONDS:
            d = C.DATA / t2_name(cond); shape = {r["stem"]: r["shape"] for r in json.load(open(d / "meta.json"))}
            imgs = [str(d / "images" / f"test_{i:05d}.png") for i in range(N_SHORT + N_LONG)]
            for r in E.syn_rows(m, imgs, test_density=cond): r["shape"] = shape[Path(r["image"]).stem]; rows.append(r)
        import pandas as pd
        D = pd.DataFrame(rows); D.to_csv(out / "test1_2.csv", index=False)
        rec = {c: {s: float((q[q["shape"] == s].max_hit >= t).mean()) for s in ("short", "long")} | {"all": float((q.max_hit >= t).mean())} for c, q in D.groupby("test_density")}
        json.dump(dict(name=name, threshold=t, recall=rec, n={"short": N_SHORT, "long": N_LONG}), open(out / "test1_2.json", "w"), ensure_ascii=False, indent=1)
        print(json.dumps(dict(name=name, threshold=t, recall={c: (round(v["short"], 3), round(v["long"], 3)) for c, v in rec.items()}), ensure_ascii=False), flush=True)
        del m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("REV1_T2_DONE", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "eval"]); ap.add_argument("--names", nargs="*"); ap.add_argument("--workers", type=int, default=8)
    A = ap.parse_args()
    if A.cmd == "build": build(A.workers)
    else: evaluate(A.names)
