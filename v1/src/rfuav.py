"""RFUAV raw IQ(100 MS/s complex float32, 중심 2.44 GHz, 1 s/파일) → 20 MHz 창 0.1 s 조각 → 스펙트로그램.
사용: python rfuav.py --device "FLYSKY FS I6X" --offset_s 0.3 --f_off 0 --out x.png
 f_off: 원래 100 MHz 창 안에서 잘라낼 20 MHz 부창의 중심 오프셋(Hz, -40e6..40e6).
"""
import argparse, sys, glob, re
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
sys.path.insert(0, str(Path(__file__).parent))
from spectrogram import spectrogram_db, to_image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "rfuav"
FS_IN, FS_OUT = 100e6, 20e6


def iq_files(device: str):
    return sorted(glob.glob(str(RAW / device / "**" / "*.iq"), recursive=True), key=lambda p: (re.findall(r"pack(\d+)", p), float(re.findall(r"_(\d+)-", p)[0]) if re.findall(r"_(\d+)-", p) else 0))


def read_chunk(path: str, offset_s: float, dur: float = 0.1) -> np.ndarray:
    """파일 내 offset_s 부터 dur 초의 complex64 IQ (메모리 맵)."""
    n0, n = int(offset_s * FS_IN), int(dur * FS_IN)
    x = np.memmap(path, dtype=np.complex64, mode="r")
    return np.array(x[n0:n0 + n])


def subband(x: np.ndarray, f_off: float = 0.0) -> np.ndarray:
    """f_off 로 주파수 이동 후 5:1 데시메이션 → 20 MS/s (폭 20 MHz)."""
    if f_off:
        t = np.arange(len(x)) / FS_IN
        x = x * np.exp(-2j * np.pi * f_off * t).astype(np.complex64)
    return resample_poly(x, 1, 5).astype(np.complex64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--device", required=True); ap.add_argument("--file_idx", type=int, default=0)
    ap.add_argument("--offset_s", type=float, default=0.0); ap.add_argument("--f_off", type=float, default=0.0); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    f = iq_files(a.device)[a.file_idx]; x = subband(read_chunk(f, a.offset_s), a.f_off)
    S, fr, t = spectrogram_db(x, FS_OUT); to_image(S).save(a.out); print(f, S.shape, "->", a.out)
