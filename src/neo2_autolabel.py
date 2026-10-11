"""Neo 2 자동 레이블 (2026-10-09, 사용자 결정 "너가 자동으로 해줘", 계획은 doc/report_261010.md 1장).
Neo 2 신호는 드론 채널(그림 가운데 10 MHz)에만 나온다. 그래서 박스 위·아래 = 드론 채널(공원 2,444.5–2,454.5, 실내 2,445.5–2,455.5 MHz,
둘 다 그림 가운데), 왼쪽·오른쪽 = 그림 안에서 신호가 처음·마지막으로 보이는 시간 칸. 사용자가 실내 ON 에 직접 그린 박스와 같은 방식이다.
시간 칸 = 학습 그림과 같은 변환의 칸(약 204 µs, 그림 한 장 97칸). 칸마다:
  band = 드론 채널 안쪽(가운데 ±4.3 MHz, 0 Hz ±150 kHz 뺌) 세기의 중앙값(잡음 대비 dB)
  guard = 드론 채널 바로 바깥(중심에서 5.2–7.5 MHz, 위·아래 따로) 세기의 중앙값 중 큰 쪽(잡음 대비 dB)
  q25 = 드론 채널 안쪽 세기의 아래쪽 25 % 값(잡음 대비 dB)
  신호 칸 = band ≥ 1 dB 이고 q25 ≥ 0.5 dB 이고 guard 가 band 보다 6 dB 이상 낮거나 1 dB 미만
           (드론 채널 경계에 걸친 작은 막대 — 정체 모름 — 가 드론으로 잡히지 않게. 막대는 바깥도 함께 세진다)
  잡음 세기: 공원은 02(드론 끔) 녹음, 실내는 실내 OFF 녹음에서 주파수 칸마다(src/park_analysis.py 와 같음).
그림 판정: 신호 칸이 3칸(약 0.6 ms) 이상이면 "보임" → 박스 하나. OFF 녹음(공원 02, 실내 OFF)은 "없음"(빈 라벨). ON 인데 신호 칸이 3칸 미만이면 "안 보임"(라벨 파일 없음).
출력: runs/neo2_labels/labels.csv(그림마다 상태·박스·신호 칸 수·세기), runs/neo2_labels/columns.npz(그림마다 칸별 band·guard),
      그리고 레이블러로 볼 수 있게 data/derived/hackrf/<같은 경로>/labels/<그림>.txt (사용자가 직접 그린 라벨 파일은 덮어쓰지 않는다)
사용: python src/neo2_autolabel.py [--write]   (--write 없으면 계산·확인만)"""
import sys, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "deploy/src"))
import hackrf_paths as HP
from iq_to_image import spectrogram_clean, N, FS, NFFT
OUT = ROOT / "runs/neo2_labels"; OUT.mkdir(parents=True, exist_ok=True); DER = ROOT / "data/derived"
F = (np.arange(NFFT) - NFFT // 2) * FS / NFFT / 1e6
BAND = (np.abs(F) <= 4.3) & (np.abs(F) >= 0.15); GLO = (F >= -7.5) & (F <= -5.2); GHI = (F >= 5.2) & (F <= 7.5); REF = (F > 5.5) & (F < 8.5)
LN2 = np.log(2); T_ON, MIN_COLS = 1.0, 3


def sessions():
    """(이름, 녹음 세션 폴더, 상태 ON/OFF, 잡음 기준 세션 폴더)"""
    out = []
    for d in sorted(HP.NEO2_PARK.iterdir()):
        if d.name[:2].isdigit() and d.name[:2] != "00": out.append((d.name, d, "OFF" if d.name.startswith("02") else "ON", HP.NEO2_PARK / "02_telescopic_OFF_near1m"))
    for k in ("ON", "OFF"): out.append(("indoor_" + k, HP.NEO2_INDOOR[k], k, HP.NEO2_INDOOR["OFF"]))
    return out


def img_dir(d): return DER / d.relative_to(ROOT / "data/raw") / "images"


def power(src, i):
    r = np.memmap(src, dtype=np.int8, mode="r"); x = r[2 * i * N:2 * (i + 1) * N].astype(np.float32); x = (x[0::2] + 1j * x[1::2]).astype(np.complex64)
    return 10 ** (spectrogram_clean(x) / 10)


def noise_of(d):
    idx = pd.read_csv(img_dir(d) / "index.csv"); cols = [power(ROOT / s, int(f)) for s, f in zip(idx.source, idx.frame)]
    return np.median(np.concatenate(cols, 1), 1) / LN2


def job(a):
    src, i, noise = a
    P = power(ROOT / src, i) / noise[:, None]; k = np.median(P[REF]) / LN2; S = P / k
    db = lambda m: 10 * np.log10(np.median(S[m], 0) / LN2)
    q25 = 10 * np.log10(np.percentile(S[BAND], 25, axis=0) / -np.log(0.75))      # 대역 아래쪽 25 % 값(잡음만 있으면 0 dB)
    return db(BAND).astype(np.float32), np.maximum(db(GLO), db(GHI)).astype(np.float32), q25.astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); A = ap.parse_args()
    rows, cols = [], {}
    noises = {}
    with Pool(18) as pool:
        for name, d, state, nd in sessions():
            if nd not in noises: noises[nd] = noise_of(nd)
            idx = pd.read_csv(img_dir(d) / "index.csv")
            res = pool.map(job, [(s, int(f), noises[nd]) for s, f in zip(idx.source, idx.frame)], chunksize=4)
            B = np.stack([r[0] for r in res]); G = np.stack([r[1] for r in res]); Q = np.stack([r[2] for r in res])
            cols[name + "__band"] = B; cols[name + "__guard"] = G; cols[name + "__q25"] = Q
            on = (B >= T_ON) & ((G < B - 6) | (G < 1.0)) & (Q >= 0.5)      # 대역 아래쪽 25 % 도 올라옴(대역 일부만 덮는 막대 제외)
            for j, (im, r) in enumerate(zip(idx.image, on)):
                c = np.flatnonzero(r); n = len(c)
                st = "none" if state == "OFF" else ("visible" if n >= MIN_COLS else "invisible")
                x0, x1 = (c[0] / 97, (c[-1] + 1) / 97) if st == "visible" else (np.nan, np.nan)
                rows.append(dict(session=name, image=im, state=st, on_cols=n, x0=x0, x1=x1, band_p90=float(np.percentile(B[j], 90)),
                                 band_on_med=float(np.median(B[j][r])) if n else np.nan, path=str((img_dir(d) / im).relative_to(ROOT))))
            print(name, state, len(idx), "장, 보임", int(sum(1 for q in rows if q["session"] == name and q["state"] == "visible")), flush=True)
    L = pd.DataFrame(rows); L.to_csv(OUT / "labels.csv", index=False); np.savez_compressed(OUT / "columns.npz", **cols)
    # 확인: OFF 녹음에서 신호 칸이 하나라도 나온 그림 수(0 이어야 기준이 잡음에 안 걸림)
    for s in ("02_telescopic_OFF_near1m", "indoor_OFF"):
        q = L[L.session == s]; print(f"확인 {s}: 신호 칸이 있는 그림 {int((q.on_cols > 0).sum())} / {len(q)}, 가장 많은 신호 칸 {int(q.on_cols.max())}")
    print(L.groupby("session").state.value_counts().unstack(fill_value=0).to_string())
    if A.write:
        n_w, n_keep = 0, 0
        for r in L.itertuples():
            lab = ROOT / r.path.replace("/images/", "/labels/"); lab = lab.with_suffix(".txt"); lab.parent.mkdir(parents=True, exist_ok=True)
            if lab.exists() and (OUT / "manual_kept.txt").exists() and str(lab) in (OUT / "manual_kept.txt").read_text(): n_keep += 1; continue
            if r.state == "invisible":
                if lab.exists(): lab.unlink()
                continue
            lab.write_text("" if r.state == "none" else f"0 {(r.x0 + r.x1) / 2:.6f} 0.500000 {r.x1 - r.x0:.6f} 0.500000\n"); n_w += 1
        print("라벨 파일 씀", n_w, "개, 사용자 라벨 그대로 둠", n_keep, "개")
    print("NEO2_AUTOLABEL_DONE")
