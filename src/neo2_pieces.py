"""v9 합성 재료 만들기 (2026-10-09 밤, 사용자 결정은 doc/report_261010.md 1·2장).
1) 공원 바탕: 공원 02(드론·조종기 끔, 3초)를 1초씩 셋으로 나눠 src/hackrf_clean.clean 으로 짧은 신호를 지운다(실내 안테나 뺀 녹음과 같은 방법).
   → data/derived/hackrf_clean/Park02_{1,2,3}.npy (1·2 = 학습 그림 바탕, 3 = 검증 그림 바탕)
2) Neo 2 조각: 자동 레이블(src/neo2_autolabel.py, runs/neo2_labels/labels.csv)에서 "보임" 그림만.
   학습용 = 공원 01·03·09, 검증용 = 공원 10. 테스트(실내 전부, 공원 13·07·04·05·06·08·11·12)와 공원 02 는 쓰지 않는다.
   조각 = 그 그림의 20 ms IQ 에서 평균을 빼고, 드론 채널(그림 가운데 ±5 MHz)만 남기고, 가운데 ±100 kHz(HackRF 의 0 Hz 줄)는 같은 쪽 바로 바깥
   100–200 kHz 의 값을 위상 그대로 옮겨 채운 것(조각을 다른 주파수로 옮길 때 0 Hz 줄이 따라가 가짜 선이 생기지 않게).
   세기 기준: 공원 02 원본(평균 뺌)의 복소 표준편차로 나눔 = 바탕(잡음 세기 1)과 같은 기준.
   → runs/neo2_pieces.npy (complex64 [조각 수, 400,000]), runs/neo2_pieces_meta.json (session, image, part, t0, t1, s0_db = 신호 칸의 드론 대역 세기 중앙값)
3) 잡음 모양: 바탕마다 주파수별 잡음 세기(평균 1 로 맞춤, 1,024칸) → runs/noise_shapes.npz (off = 안테나 뺀 녹음 01–03, park = 공원 02)
사용: python src/neo2_pieces.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import hackrf_paths as HP, hackrf_clean as HC, synth_iq as S
N, FS = S.N, S.FS
TRAIN, VAL = ("01", "03", "09"), ("10",)


def park02():
    f = sorted((HP.NEO2_PARK / "02_telescopic_OFF_near1m/captures").glob("*/iq.sigmf-data"))[0]
    r = np.fromfile(f, dtype=np.int8).astype(np.float32); return (r[0::2] + 1j * r[1::2]).astype(np.complex64)


def shape(x, nf=1024):
    x = x - x.mean(); X = x[: len(x) // nf * nf].reshape(-1, nf) * np.hanning(nf)
    P = np.median(np.abs(np.fft.fftshift(np.fft.fft(X, axis=1), axes=1)) ** 2, 0); return (P / P.mean()).astype(np.float64)


if __name__ == "__main__":
    out = ROOT / "data/derived/hackrf_clean"; rng = np.random.default_rng(7)
    x = park02(); sd = float(np.sqrt(np.mean(np.abs(x - x.mean()) ** 2))); print("공원 02 길이", len(x) / FS, "초, 복소 표준편차", round(sd, 4))
    for k in range(3):
        seg = x[k * int(FS): (k + 1) * int(FS)]; y, frac, pdb, miss = HC.clean(seg, rng); np.save(out / f"Park02_{k + 1}.npy", y)
        print(f"Park02_{k + 1}: {len(seg) / FS:.3f} 초, 지운 칸 {frac * 100:.2f} %, 채운 자리 세기 차 {pdb:+.2f} dB, 못 옮긴 덩어리 {miss}")
    sh = dict(off=np.mean([shape(np.load(out / f"AntennaOff_{k}.npy")) for k in ("01", "02", "03")], 0), park=shape(np.load(out / "Park02_1.npy")))
    np.savez(ROOT / "runs/noise_shapes.npz", f_mhz=(np.arange(1024) - 512) * FS / 1024 / 1e6, **sh)
    print("잡음 모양(평균 1 기준 dB): 가운데 ±1 MHz / 끝 ±9 MHz 밖 —", {k: (round(10 * np.log10(v[506:518].mean()), 2), round(10 * np.log10(np.r_[v[:52], v[-52:]].mean()), 2)) for k, v in sh.items()})
    L = pd.read_csv(ROOT / "runs/neo2_labels/labels.csv"); L["k"] = L.session.str[:2]
    use = L[(L.state == "visible") & L.k.isin(TRAIN + VAL)].reset_index(drop=True)
    fr = np.fft.fftfreq(N, 1 / FS); band = np.abs(fr) <= 5e6; dc = np.abs(fr) < 100e3
    K = int(round(100e3 / (FS / N))); pos = np.arange(0, K); neg = np.arange(N - K, N)     # 10-09: 처음엔 위상을 무작위로 채워 신호가 시간 전체로 번져 가로선이 생겼음
    arr = np.lib.format.open_memmap(ROOT / "runs/neo2_pieces.npy", mode="w+", dtype=np.complex64, shape=(len(use), N)); meta = []
    idx = {}
    for i, r in use.iterrows():
        d = ROOT / Path(r.path).parent.parent
        if d not in idx: idx[d] = pd.read_csv(d / "index.csv").set_index("image")          # d = …/images
        q = idx[d].loc[r.image]; m = np.memmap(ROOT / q.source, dtype=np.int8, mode="r"); j = int(q.frame)
        z = m[2 * j * N: 2 * (j + 1) * N].astype(np.float32); z = (z[0::2] + 1j * z[1::2]).astype(np.complex64); z -= z.mean()
        F = np.fft.fft(z); F[~band] = 0
        F[pos] = F[pos + K]; F[neg] = F[neg - K]          # 0 Hz ±100 kHz 를 같은 쪽 100 kHz 바깥 값으로(위상 그대로 → 켜짐·꺼짐 시간 모양 유지)
        arr[i] = (np.fft.ifft(F) / sd).astype(np.complex64)
        meta.append(dict(session=r.session, image=r.image, part="train" if r.k in TRAIN else "val", t0=float(r.x0) * S.WIN, t1=float(r.x1) * S.WIN, s0_db=float(r.band_on_med)))
    arr.flush(); del arr; json.dump(meta, open(ROOT / "runs/neo2_pieces_meta.json", "w"), ensure_ascii=False, indent=0)
    M = pd.DataFrame(meta); print(M.groupby(["part", "session"]).s0_db.describe()[["count", "min", "50%", "max"]].round(2).to_string())
    print("NEO2_PIECES_DONE")
