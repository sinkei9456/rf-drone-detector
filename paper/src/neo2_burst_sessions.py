"""세션별 Neo 2 버스트 길이·반복 간격 측정 (2026-11-12). 부록 C 표 C-1.
대상: 공원 세션 01–13(data/raw/hackrf/3_neo2_park, 중심 2449.5 MHz), 실내 Neo 2 ON(data/raw/hackrf/2_neo2_indoor/20261008_drone_on, 중심 2450.5 MHz).
방법: 100 ms 간격으로 20 ms 구간을 잘라, 평균을 빼고 수신기 DC(±100 kHz)를 지운 뒤 Neo 2 채널(가운데 ±5 MHz)만 남긴다.
  채널 전력을 1024 샘플(51 µs)씩 이동 평균하고, 하위 10 %(꺼짐)와 상위 10 %(켜짐)의 가운데를 넘는 구간을 버스트로 본다.
  0.1 ms 보다 짧은 틈은 잇고 0.05 ms 보다 짧은 버스트는 버린다.
  켜짐·꺼짐 세기 차이(상위 10 % ÷ 하위 10 %)가 1 dB 미만인 구간은 잴 수 없음으로 둔다. 구간 경계에 걸린 버스트는 길이에서 뺀다.
출력: paper/runs/neo2_bursts/{sessions.csv, summary.md}
사용: python paper/src/neo2_burst_sessions.py"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "paper/runs/neo2_bursts"
FS, N = 20e6, 400_000; FR = np.fft.fftfreq(N, 1 / FS)


def read(f):
    r = np.fromfile(f, dtype=np.int8).astype(np.float32); x = (r[0::2] + 1j * r[1::2]).astype(np.complex64); return x - x.mean()


def bursts(y):
    e = np.convolve(np.abs(y) ** 2, np.ones(1024) / 1024, "same"); lo, hi = np.quantile(e, 0.1), np.quantile(e, 0.9); d_db = 10 * np.log10(hi / lo)
    if d_db < 1.0: return None, d_db
    on = e > (lo + hi) / 2; d = np.diff(np.r_[0, on.astype(int), 0]); out = []
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if out and a - out[-1][1] < 0.1e-3 * FS: out[-1][1] = b
        else: out.append([a, b])
    return [(a, b) for a, b in out if b - a >= 0.05e-3 * FS], d_db


def session(name, files, f_ch):
    on, per, dd, n_seg, n_ok = [], [], [], 0, 0
    for f, fc in files:
        x = read(f); off = f_ch - fc
        for s in range(0, len(x) - N, int(0.1 * FS)):
            F = np.fft.fft(x[s:s + N]); F[np.abs(FR) <= 100e3] = 0
            y = np.fft.ifft(F) * np.exp(-2j * np.pi * off * np.arange(N) / FS); G = np.fft.fft(y); G[np.abs(FR) > 5e6] = 0
            b, d_db = bursts(np.fft.ifft(G)); n_seg += 1; dd.append(d_db)
            if not b: continue
            n_ok += 1; on += [(e - a) / FS * 1e3 for a, e in b if a > 0 and e < N]; per += list(np.diff([a / FS * 1e3 for a, e in b]))
    q = lambda v, p: round(float(np.percentile(v, p)), 2) if v else None
    return dict(session=name, n_files=len(files), n_segments=n_seg, n_measured=n_ok, n_bursts=len(on), on_ms_median=q(on, 50), on_ms_q25=q(on, 25),
                on_ms_q75=q(on, 75), period_ms_median=q(per, 50), on_off_db_median=round(float(np.median(dd)), 1) if dd else None)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True); rows = []
    for d in sorted((ROOT / "data/raw/hackrf/3_neo2_park").glob("[01][0-9]_*")):
        fs = sorted(glob.glob(f"{d}/**/iq.sigmf-data", recursive=True))
        files = [(f, json.load(open(f.replace(".sigmf-data", ".sigmf-meta")))["captures"][0]["core:frequency"]) for f in fs]
        rows.append(session(d.name, files, 2449.5e6)); print(rows[-1], flush=True)
    fs = sorted(glob.glob(str(ROOT / "data/raw/hackrf/2_neo2_indoor/20261008_drone_on/recordings/*/*.cs8")))
    rows.append(session("indoor_ON_20261008", [(f, 2450.5e6) for f in fs], 2450.5e6)); print(rows[-1], flush=True)
    D = pd.DataFrame(rows); D.to_csv(OUT / "sessions.csv", index=False)
    (OUT / "summary.md").write_text("# 세션별 Neo 2 버스트\n\npaper/src/neo2_burst_sessions.py 가 만든 표(길이·간격 ms, 세기 차이 dB).\n\n" + D.to_markdown(index=False) + "\n", encoding="utf-8")
    print("NEO2_BURSTS_DONE")
