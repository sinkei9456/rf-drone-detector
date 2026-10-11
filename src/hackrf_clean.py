"""HackRF 바탕 다듬기 (2026-10-07, 새 방법): 팀원의 HackRF 빈 녹음 중 **안테나 뺀 녹음 5개**(1초씩)에서 블루투스로 보이는 덩어리를 지운다.
(사용자 결정 10-07: 바탕은 안테나 뺀 녹음만. 안테나 끼운 녹음 5개는 지우지 않은 그대로 실측 테스트에만 쓴다.)
라벨을 만드는 일이 아니다. 덜 지우면 라벨 없는 블루투스가 조금 남을 뿐이고, 잡음을 지우면 잡음이 잡음으로 바뀔 뿐이다.
방법(녹음 1초 전체를 한 번에):
  1) 평균(DC)을 빼고 단시간 푸리에 변환(1,024점, 256점 간격, 주파수 칸 19.5 kHz, 시간 칸 12.8 µs).
  2) 주파수마다 바닥 세기 = 그 주파수 칸들의 세기 중앙값 ÷ ln2 (잡음만 있으면 평균과 같음).
  3) 지울 칸: 주파수 5칸(98 kHz) × 시간 9칸(115 µs) 평균 세기가 바닥보다 5 dB 이상 센 칸들이 이어진 덩어리 중,
     높이 3 MHz 이하이고 50칸 이상인 것(블루투스 모양). 그 둘레(주파수 ±5칸 = ±98 kHz, 시간 ±10칸 = ±0.13 ms)까지 넓혀 지운다.
     (10-07 첫 시도는 한 칸만 튀는 것도 잡아 둘레까지 지우는 바람에 29 %를 지웠다. 그림의 세로 선처럼 아주 짧게 넓은 주파수에 걸친 것은
      이 수신기의 실제 바탕이고 블루투스가 아니므로 남긴다.)
     바닥 세기는 바닥의 10배 넘는 칸을 뺀 평균으로 다시 잰다.
  4) 지울 덩어리마다, 같은 주파수 줄의 다른 시각에서 지울 칸이 하나도 없는 같은 크기 영역을 찾아 그 값을 통째로 옮겨 채우고 다시 IQ 로 되돌린다.
     (10-07 첫 시도처럼 칸마다 따로 뽑은 잡음으로 채우면, 되돌릴 때 겹친 칸끼리 서로 상쇄돼 지운 자리가 주변보다 어둡게 남았다.)
확인: 지우기 전·후 20 ms 그림마다 바닥보다 10 dB 이상 센 덩어리 수(runs/review/hackrf_blobs.txt 와 같은 방법), 주파수별 바닥 세기 변화.
출력: data/derived/hackrf_clean/<상태>_<번호>.npy (complex64, 20 MS/s, 1초), runs/review/hackrf_clean.txt, doc/fig_report/hackrf_clean.png"""
import sys, glob, json
from pathlib import Path
import numpy as np
from scipy.signal import stft, istft
from scipy import ndimage
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import sys as _sys; _sys.path.insert(0, str(ROOT / "src")); import hackrf_paths as HP   # 10-09 HackRF 폴더 정리: 녹음 경로는 hackrf_paths 에서
import synth_iq as S
from spectrogram import spectrogram_db
OUT = ROOT / "data/derived/hackrf_clean"; OUT.mkdir(parents=True, exist_ok=True)
COND = ("AntennaOff",); NP, HOP = 1024, 256


def load(c, fid):
    f = [str(p) for p in HP.nodrone_files(c, fid)]; assert len(f) == 1, (c, fid)
    r = np.fromfile(f[0], dtype=np.int8).astype(np.float32); return (r[0::2] + 1j * r[1::2]).astype(np.complex64)


def clean(x, rng):
    x = x - x.mean()
    _, _, Z = stft(x, fs=S.FS, window="hann", nperseg=NP, noverlap=NP - HOP, return_onesided=False, boundary="zeros", padded=True)
    P = np.abs(Z) ** 2; f0 = np.median(P, axis=1, keepdims=True) / np.log(2)
    floor = np.array([[P[i][P[i] < 10 * f0[i, 0]].mean()] for i in range(P.shape[0])])          # 센 칸을 뺀 바닥 세기
    Rs = ndimage.uniform_filter(P / floor, size=(5, 9))
    lab, _ = ndimage.label(Rs > 10 ** 0.5); m = np.zeros(P.shape, dtype=bool); df = S.FS / NP
    for j, sl in enumerate(ndimage.find_objects(lab), 1):
        if (sl[0].stop - sl[0].start) * df <= 3e6:
            mm = lab[sl] == j
            if mm.sum() >= 50: m[sl] |= mm
    m = ndimage.binary_dilation(m, structure=np.ones((11, 21), dtype=bool))
    Z0 = Z; Z = Z.copy(); T = Z.shape[1]; lab2, _ = ndimage.label(m); miss = 0
    for j, sl in enumerate(ndimage.find_objects(lab2), 1):
        r0, r1, c0, c1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop; w = c1 - c0; mm = lab2[sl] == j
        for _ in range(500):
            s0 = int(rng.integers(0, T - w))
            if abs(s0 - c0) >= w and not m[r0:r1, s0:s0 + w].any(): break
        else:
            miss += 1; continue
        Z[r0:r1, c0:c1][mm] = Z0[r0:r1, s0:s0 + w][mm]
    _, y = istft(Z, fs=S.FS, window="hann", nperseg=NP, noverlap=NP - HOP, input_onesided=False, boundary=True)
    y = y[:len(x)].astype(np.complex64)
    _, _, Zy = stft(y, fs=S.FS, window="hann", nperseg=NP, noverlap=NP - HOP, return_onesided=False, boundary="zeros", padded=True)
    Py = np.abs(Zy) ** 2; rows = m.any(axis=1)
    patch_db = 10 * np.log10(Py[m].mean() / Py[rows][~m[rows]].mean())                       # 채운 자리 세기 ÷ 같은 줄의 나머지 세기
    return y, float(m.mean()), float(patch_db), miss


def blobs(x):
    """20 ms 그림 50장에서 바닥보다 10 dB 이상 센 덩어리(3칸 이상) 수. 주파수 156 kHz 칸, 바닥 = 그 녹음 전체 중앙값."""
    A = np.stack([10 * np.log10((10 ** (S.spectrogram_v7(x[i * S.N:(i + 1) * S.N]) / 10)).reshape(128, 64, -1).mean(1)) for i in range(50)])
    ex = A - np.median(A, axis=(0, 2), keepdims=True); per = []
    for i in range(50):
        lab, _ = ndimage.label(ex[i] > 10); per.append(sum(1 for j, sl in enumerate(ndimage.find_objects(lab), 1) if (lab[sl] == j).sum() >= 3))
    return np.array(per), np.median(A, axis=(0, 2))


if __name__ == "__main__":
    rng = np.random.default_rng(7); L = ["HackRF 바탕 다듬기(블루투스 지우기) 결과 — 20 ms 그림 50장마다 바닥보다 10 dB 이상 센 덩어리 수 (가운데 줄은 두 경우 모두 지운 뒤 셈)"]
    ex = {}
    for c in COND:
        for fid in ("01", "02", "03", "04", "05"):
            x = load(c, fid); y, frac, pdb, miss = clean(x, rng); np.save(OUT / f"{c}_{fid}.npy", y)
            b0, f0 = blobs(x); b1, f1 = blobs(y)
            L.append(f"{c} {fid}번째 녹음: 지운 칸 {frac * 100:.1f} % (옮겨 올 곳을 못 찾은 덩어리 {miss}개), 채운 자리 세기 − 같은 주파수의 나머지 {pdb:+.2f} dB | 덩어리 있는 그림 {int((b0 > 0).sum())}/50 → {int((b1 > 0).sum())}/50, 그림당 {b0.mean():.2f} → {b1.mean():.2f}개 | "
                     f"주파수별 바닥 세기 변화 중앙값 {np.median(f1 - f0):+.2f} dB (가장 큰 변화 {np.max(np.abs(f1 - f0)):.2f} dB)")
            if fid in ("01", "03", "04", "05"): ex[(c, fid)] = (x, y)
    (ROOT / "runs/review/hackrf_clean.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 6, figsize=(16, 6.2))
    cols = [("AntennaOff", "01", 10), ("AntennaOff", "01", 40), ("AntennaOff", "03", 25), ("AntennaOff", "04", 10), ("AntennaOff", "04", 40), ("AntennaOff", "05", 25)]
    for j, (c, fid, k) in enumerate(cols[:6]):
        x, y = ex[(c, fid)]
        for i, (z, lab) in enumerate(((x, "before (DC line removed)"), (y, "after Bluetooth removal"))):
            a = ax[i][j]; a.imshow(S.to_gray_v7(z[k * S.N:(k + 1) * S.N]), cmap="gray", vmin=0, vmax=255); a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{'antenna OFF' if c == 'AntennaOff' else 'antenna ON'}, recording {fid}, frame {k}\n{lab}", fontsize=8.5)
    fig.tight_layout(); fig.savefig(ROOT / "doc/fig_report/hackrf_clean.png", dpi=55); print("그림 doc/fig_report/hackrf_clean.png")
