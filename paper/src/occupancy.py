"""스펙트럼 점유율(spectrum occupancy) 정의. 원고 3.3절의 간섭 밀도와 보충 자료 S6의 혼잡 장소 점유율이 모두 이 계산을 쓴다.
20 ms IQ(20 MS/s)를 학습 이미지와 같은 STFT(8,192점 Hann, 간격 4,081, 시간 97칸)로 바꾸고, 주파수 16칸씩 묶어 512줄(39 kHz)로 만든다.
줄마다 잡음 세기(하위 10 % 세기)보다 6 dB 이상 센 칸의 비율이 점유율이다. 가운데 ±100 kHz(DC) 줄과 드론 박스 칸은 뺀다.
(이 코드는 원래 실험 A 모듈의 power·occ·read_iq 와 그림 모듈의 occ_parts 를 그대로 옮긴 것이다.)"""
import numpy as np
FS, N, NFFT, HOP, NF = 20e6, 400_000, 8192, 4081, 512
WIN = np.hanning(NFFT).astype(np.float32); FR = (np.arange(NF) - NF / 2 + 0.5) * FS / NF; DC = np.abs(FR) <= 100e3; THR_DB = 6.0


def power(x):
    """20 ms IQ → (97 시간 칸, 512 주파수 줄) 세기(선형). 줄은 낮은 주파수→높은 주파수."""
    fr = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP][:97] * WIN
    P = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=1), axes=1)) ** 2
    return P.reshape(97, NF, NFFT // NF).mean(-1)


def occ(P, floor, mask=None):
    on = P > floor[None, :] * 10 ** (THR_DB / 10); keep = np.broadcast_to(~DC[None, :], P.shape).copy()
    if mask is not None: keep &= ~mask
    return float(on[keep].mean()), float(keep.mean())


def read_iq(f):
    r = np.fromfile(f, dtype=np.int8).astype(np.float32); x = (r[0::2] + 1j * r[1::2]).astype(np.complex64)
    return [x[i:i + N] for i in range(0, len(x) - N + 1, N)]


def occ_parts(P, floor, drone_boxes=()):
    """드론 박스(YOLO 좌표 xc, yc, w, h)를 뺀 점유율. 반환 (센 칸, 잰 칸, 점유율)."""
    on = P > floor[None, :] * 10 ** (THR_DB / 10); keep = np.broadcast_to(~DC[None, :], P.shape).copy()
    for xc, yc, w, h in drone_boxes:
        t0, t1 = int(np.floor((xc - w / 2) * 97)), int(np.ceil((xc + w / 2) * 97)); f0, f1 = int(np.floor((1 - (yc + h / 2)) * NF)), int(np.ceil((1 - (yc - h / 2)) * NF))
        keep[max(t0, 0):t1, max(f0, 0):f1] = False
    return on, keep, float(on[keep].mean())
