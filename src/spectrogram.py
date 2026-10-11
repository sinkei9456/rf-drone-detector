"""IQ(cf32) -> 스펙트로그램(PNG) 변환기.

AI Hub 71954 .npy(cf32_le, 20 MHz, 0.1 s)와 팀 SDR 캡처를 같은 함수로 처리한다.
사용:  python spectrogram.py <iq.npy|iq.cf32> [--fs 20e6] [--out out.png]
자체 검증: python spectrogram.py --selftest
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.signal import stft
from PIL import Image


def load_iq(path: str | Path) -> np.ndarray:
    """.npy(complex64 또는 [N,2] float32) 또는 raw .cf32 를 complex64 로 읽는다."""
    path = Path(path)
    if path.suffix == ".npy":
        x = np.load(path, mmap_mode="r")
        if np.iscomplexobj(x):
            return np.asarray(x, dtype=np.complex64)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2 and x.shape[1] == 2:          # [N,2] I,Q
            return (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)
        return x.view(np.complex64)                  # interleaved 1-D
    raw = np.fromfile(path, dtype=np.float32)
    return raw.view(np.complex64)


def spectrogram_db(iq: np.ndarray, fs: float, nfft: int = 1024, hop: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """반환: S_db[freq, time], f(Hz, 중심 0 기준, 오름차순), t(s)."""
    f, t, Z = stft(iq, fs=fs, nperseg=nfft, noverlap=nfft - hop, window="hann",
                   return_onesided=False, boundary=None, padded=False)
    f = np.fft.fftshift(f); Z = np.fft.fftshift(Z, axes=0)
    S = 20 * np.log10(np.abs(Z) + 1e-12)
    return S.astype(np.float32), f, t


def to_image(S_db: np.ndarray, size: tuple[int, int] = (640, 640),
             p_lo: float = 1.0, p_hi: float = 99.5) -> Image.Image:
    """백분위 정규화 후 0~255 그레이스케일. 위가 고주파, 왼쪽이 시작 시각."""
    lo, hi = np.percentile(S_db, [p_lo, p_hi])
    g = np.clip((S_db - lo) / max(hi - lo, 1e-6), 0, 1)
    img = Image.fromarray((g[::-1] * 255).astype(np.uint8))   # 고주파를 위로
    return img.resize(size, Image.BILINEAR)


def iq_to_png(iq_path, fs, out_png, **kw):
    iq = load_iq(iq_path)
    S, f, t = spectrogram_db(iq, fs, **kw)
    to_image(S).save(out_png)
    return S.shape, float(t[-1])


# ---------------- 자체 검증용 합성 신호 ----------------
def bandlimited_burst(n: int, bw: float, fs: float, rng, rms: float = 0.5) -> np.ndarray:
    """폭 bw 의 대역 제한 잡음 버스트(OFDM/GFSK 덩어리 흉내), RMS 정규화. 중심 0 Hz."""
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    F = np.fft.fft(noise); fr = np.fft.fftfreq(n, 1 / fs)
    F[np.abs(fr) > bw / 2] = 0
    b = np.fft.ifft(F)
    return (b / np.sqrt(np.mean(np.abs(b) ** 2) + 1e-12) * rms).astype(np.complex64)


def synth_scene(fs=20e6, dur=0.1, seed=0):
    """잡음 + OFDM형 광대역 버스트(드론/Wi-Fi 흉내) + 협대역 FHSS 홉(블루투스 흉내).
    반환 boxes: (label, t0, t1, f_lo, f_hi)"""
    rng = np.random.default_rng(seed)
    n = int(fs * dur); t = np.arange(n) / fs
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * 0.05
    boxes = []
    bw, fc = 10e6, 3e6                                   # 10 MHz 폭, 2 ms 주기 1 ms 켜짐
    for k in range(int(dur / 2e-3)):
        s0, s1 = int(k * 2e-3 * fs), int((k * 2e-3 + 1e-3) * fs)
        x[s0:s1] += bandlimited_burst(s1 - s0, bw, fs, rng, 0.5) * np.exp(2j * np.pi * fc * t[s0:s1])
        boxes.append(("ofdm", s0 / fs, s1 / fs, fc - bw / 2, fc + bw / 2))
    for k in range(int(dur / 0.6e-3)):                   # 1 MHz 폭 홉, 0.6 ms 마다 0.4 ms
        s0, s1 = int(k * 0.6e-3 * fs), int((k * 0.6e-3 + 0.4e-3) * fs)
        f0 = rng.uniform(-9e6, -4e6)
        x[s0:s1] += bandlimited_burst(s1 - s0, 1e6, fs, rng, 0.3) * np.exp(2j * np.pi * f0 * t[s0:s1])
        boxes.append(("hop", s0 / fs, s1 / fs, f0 - 0.5e6, f0 + 0.5e6))
    return x, boxes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("iq", nargs="?")
    ap.add_argument("--fs", type=float, default=20e6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        out = Path(__file__).resolve().parents[1] / "runs" / "selftest"
        out.mkdir(parents=True, exist_ok=True)
        x, boxes = synth_scene()
        np.save(out / "synth.npy", x)
        shape, T = iq_to_png(out / "synth.npy", 20e6, out / "synth.png", nfft=a.nfft, hop=a.hop)
        json.dump(boxes, open(out / "synth_boxes.json", "w"))
        print("STFT shape", shape, "duration s", round(T, 4), "->", out / "synth.png")
    else:
        out = a.out or str(Path(a.iq).with_suffix(".png"))
        print(iq_to_png(a.iq, a.fs, out, nfft=a.nfft, hop=a.hop), "->", out)
