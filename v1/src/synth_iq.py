"""IQ 단계 합성 (2026-10-03). 바탕(잡음 + 와이파이·블루투스 신호)과 드론 신호 조각을 모두 IQ 로 만들어 더한 뒤 한 번에 스펙트로그램을 만든다.
그림 한 장 = 20 MHz × 20 ms (HackRF 와 같은 조건), 20 MS/s, 400,000 표본. 모든 주파수는 창 가운데(0 Hz) 기준.
- 바탕: 수신기 잡음(가우스) + 와이파이 신호(폭 16.6 MHz, 0.1~3 ms, 2.4 GHz 와이파이 채널 1~13 자리) + 블루투스 신호(클래식 1 MHz × 0.4 ms, BLE 2 MHz × 0.1~0.4 ms).
  개수(n_wifi, n_bt)로 붐비는 정도를 조절한다. 어디에 넣었는지 알므로 박스가 정확하다.
- 드론 조각: 드론만 녹음한 IQ(Tampere 무반사실, RFUAV)에서 자동 규칙으로 넓은 링크를 찾아 그 띠만 잘라 낸다(가운데를 0 Hz 로 옮기고 띠 폭만 남김).
  **누출 방지: Tampere 는 학습 구간(0~0.56초)만 쓴다(검증 0.6~0.76초, 테스트 0.8~1.0초는 안 씀). RFUAV 는 어느 테스트셋에도 없다.**
- 세기: 신호의 1 Hz 당 세기가 잡음 바닥보다 snr_db 만큼 크게 맞춘다(그림에서 보이는 밝기 차이와 같은 뜻).
반환 박스: (종류, t0, t1, f_lo, f_hi) — 시간은 초, 주파수는 창 가운데 기준 Hz."""
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from spectrogram import spectrogram_db, to_image
import linklabel

FS, WIN = 20e6, 0.02; N = int(FS * WIN); PX = 640
NFFT, HOP = 8192, int(FS * 0.1 / 490)                    # 우리 그림 격자(주파수 칸 2.44 kHz, 시간 칸 204 µs, 97칸)
LINK_MAX = 18e6                                          # 조각 폭 상한. v2(10-04) 18 MHz, v3(10-06, 사용자 결정) 20 MHz — synth_iq_build --ver v3 가 바꿈
F_WIFI = [(2.412e9 + 5e6 * (c - 1)) - 2.44e9 for c in range(1, 14)]   # 창 가운데를 2.44 GHz 로 봤을 때 와이파이 채널 1~13 의 자리


def band_burst(n, f_lo, f_hi, rng, rms):
    """[f_lo, f_hi] 안에만 성분이 있는 잡음 덩어리(와이파이·블루투스 흉내). 창 밖 부분은 잘린다(감아 들어오지 않음)."""
    f_lo, f_hi = max(f_lo, -FS / 2), min(f_hi, FS / 2)
    if f_hi - f_lo < 20e3: return np.zeros(n, np.complex64)
    F = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n)); fr = np.fft.fftfreq(n, 1 / FS)
    F[(fr < f_lo) | (fr > f_hi)] = 0; b = np.fft.ifft(F)
    return (b / np.sqrt(np.mean(np.abs(b) ** 2) + 1e-20) * rms).astype(np.complex64)


def _rms_for(snr_db, bw, floor=1.0):
    """1 Hz 당 세기가 바닥(전력 floor, 폭 FS)보다 snr_db 크도록 하는 덩어리 전체 rms."""
    return float(np.sqrt(10 ** (snr_db / 10) * floor / FS * bw))


def make_background(rng, n_wifi, n_bt, floor=1.0, snr_wifi=(5, 25), snr_bt=(8, 30), base=None):
    """잡음 + 와이파이 n_wifi 개 + 블루투스 n_bt 개(모두 창 안에 보이는 것만 셈). 반환 (iq, 박스).
    base: 바탕 잡음으로 쓸 실제 녹음 20 ms(잡음 세기 1 로 맞춘 것, 10-07 v6). 없으면 고른 잡음을 만든다(v2~v5 와 같음)."""
    if base is not None: x = (np.asarray(base, dtype=np.complex64) * np.float32(np.sqrt(floor))).copy(); boxes = []
    else: x = ((rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.sqrt(floor / 2)).astype(np.complex64); boxes = []
    for _ in range(n_wifi):
        fo = F_WIFI[int(rng.integers(len(F_WIFI)))]; bw = 16.6e6
        if fo + bw / 2 < -FS / 2 + 1e6 or fo - bw / 2 > FS / 2 - 1e6: continue          # 창에 거의 안 들어오는 채널
        s0 = int(rng.uniform(-1e-3, WIN) * FS); s1 = s0 + int(rng.uniform(0.1e-3, 3e-3) * FS); s0, s1 = max(0, s0), min(N, s1)
        if s1 - s0 < 100: continue
        x[s0:s1] += band_burst(s1 - s0, fo - bw / 2, fo + bw / 2, rng, _rms_for(rng.uniform(*snr_wifi), bw, floor))
        boxes.append(("wifi", s0 / FS, s1 / FS, max(fo - bw / 2, -FS / 2), min(fo + bw / 2, FS / 2)))
    for _ in range(n_bt):
        ble = rng.random() < 0.3; bw = 2e6 if ble else 1e6; dur = rng.uniform(0.1e-3, 0.4e-3) if ble else 0.4e-3
        fo = rng.uniform(-FS / 2 + bw, FS / 2 - bw); s0 = int(rng.uniform(0, WIN - dur) * FS); s1 = s0 + int(dur * FS)
        x[s0:s1] += band_burst(s1 - s0, fo - bw / 2, fo + bw / 2, rng, _rms_for(rng.uniform(*snr_bt), bw, floor))
        boxes.append(("bt", s0 / FS, s1 / FS, fo - bw / 2, fo + bw / 2))
    return x, boxes


VIS_WIFI, VIS_BT_HZ, VIS_DRONE = 0.20, 0.5e6, 0.50         # 10-07 v8: 그림(세로 20 MHz) 안에 보이는 최소 비율·높이 (사용자 승인)


def make_background_v8(rng, n_wifi, n_bt, floor=1.0, snr_wifi=(5, 25), snr_bt=(8, 30), base=None):
    """10-07 v8: 와이파이·블루투스를 그림 가장자리에서 잘린 것까지 넣는다(시연은 폭 20 MHz 를 고정하고 중심 주파수를 바꿔 가며 보므로).
    와이파이(폭 16.6 MHz): 가운데 위치를 고르게 뽑되 그림 안에 폭의 20 % 이상 보이게. 블루투스(1 MHz·0.4 ms 또는 2 MHz·0.1~0.4 ms): 0.5 MHz 이상 보이게.
    잘린 덩어리도 1 Hz 당 세기는 통째일 때와 같게(보이는 폭만큼만 전체 세기를 줌). 박스는 그림 안에 보이는 부분만. 반환 (iq, 박스, 보이는 비율 목록)."""
    if base is not None: x = (np.asarray(base, dtype=np.complex64) * np.float32(np.sqrt(floor))).copy()
    else: x = ((rng.standard_normal(N) + 1j * rng.standard_normal(N)) * np.sqrt(floor / 2)).astype(np.complex64)
    boxes, vis = [], []
    def put(kind, bw, vis_min_hz, s0, s1, snr):
        fo = float(rng.uniform(-FS / 2 - bw / 2 + vis_min_hz, FS / 2 + bw / 2 - vis_min_hz))
        lo, hi = max(fo - bw / 2, -FS / 2), min(fo + bw / 2, FS / 2); v = hi - lo
        x[s0:s1] += band_burst(s1 - s0, lo, hi, rng, _rms_for(snr, bw, floor) * np.sqrt(v / bw))
        boxes.append((kind, s0 / FS, s1 / FS, lo, hi)); vis.append((kind, round(v / bw, 3)))
    for _ in range(n_wifi):
        s0 = int(rng.uniform(-1e-3, WIN) * FS); s1 = s0 + int(rng.uniform(0.1e-3, 3e-3) * FS); s0, s1 = max(0, s0), min(N, s1)
        snr = rng.uniform(*snr_wifi)
        if s1 - s0 < 100: continue
        put("wifi", 16.6e6, VIS_WIFI * 16.6e6, s0, s1, snr)
    for _ in range(n_bt):
        ble = rng.random() < 0.3; bw = 2e6 if ble else 1e6; dur = rng.uniform(0.1e-3, 0.4e-3) if ble else 0.4e-3
        s0 = int(rng.uniform(0, WIN - dur) * FS); s1 = s0 + int(dur * FS)
        put("bt", bw, VIS_BT_HZ, s0, s1, rng.uniform(*snr_bt))
    return x, boxes, vis


def place_drone_v8(rng, bw, p_full=0.3):
    """10-07 v8: 드론 조각의 가운데 위치 fo. 30 %는 그림 안에 통째로(가능하면, 위아래 0.5 MHz 여유), 나머지는 가운데가 그림 안 아무 데나
    (= 폭의 50 % 이상이 보임). 반환 fo(Hz)."""
    if rng.random() < p_full and bw <= FS - 1e6: return float(rng.uniform(-FS / 2 + bw / 2 + 0.5e6, FS / 2 - bw / 2 - 0.5e6))
    return float(rng.uniform(-FS / 2, FS / 2))


def cut_to_window(iq, fo):
    """조각을 fo 로 옮겼을 때 그림(±FS/2) 밖으로 나갈 성분을 미리 지운다(감겨 들어오지 않게). 1 Hz 당 세기는 그대로."""
    fr = np.fft.fftfreq(len(iq), 1 / FS); F = np.fft.fft(iq)
    F[~((fr + fo <= FS / 2 - 0.05e6) & (fr + fo >= -FS / 2 + 0.05e6))] = 0
    return np.fft.ifft(F).astype(np.complex64)


# ---------------- 드론 조각 ----------------
def find_wide_link(x, fs):
    """드론만 있는 IQ 에서 자동 규칙으로 넓은 링크(≥4 MHz)를 찾는다. 반환 (fmid, bw, t0, t1) 또는 None. 주파수는 창 가운데 기준."""
    S, _, _ = spectrogram_db(x, fs, nfft=8192, hop=int(fs * 0.1 / 490)); P = 10 ** (S.astype(np.float64) / 10)
    links, _, _ = linklabel.label_power(P, fs / P.shape[0], WIN / P.shape[1])
    wide = [L for L in links if L["cls"] == 0]
    if not wide: return None
    fit = [L for L in wide if L["f1"] - L["f0"] <= LINK_MAX] or wide          # 20 MHz 창에 들어가는 링크를 우선(폭 × 길이가 가장 큰 것)
    L = max(fit, key=lambda L: (L["f1"] - L["f0"]) * (L["t1"] - L["t0"]))
    return (L["f0"] + L["f1"]) / 2, L["f1"] - L["f0"], L["t0"], L["t1"]


def cut_link(x, fs, fmid, bw, margin=1.2):
    """링크 가운데를 0 Hz 로 옮기고 20 MS/s 로 줄인 뒤 띠 폭(×margin)만 남긴다. 반환 (20 ms·20 MS/s 조각, 실제 차지한 폭).
    자동 규칙의 폭은 실제보다 넓을 때가 있어(이웃 신호와 묶임), 잘라 낸 뒤 스펙트럼에서 봉우리보다 15 dB 안에 드는 범위를 실제 폭으로 다시 재고
    그 가운데를 0 Hz 로 다시 맞춘다."""
    t = np.arange(len(x)) / fs; y = (x * np.exp(-2j * np.pi * fmid * t)).astype(np.complex64)
    y = resample_poly(y, 1, int(round(fs / FS))).astype(np.complex64)[:N]
    if len(y) < N: y = np.pad(y, (0, N - len(y)))
    fr = np.fft.fftfreq(N, 1 / FS)
    F = np.fft.fft(y); F[np.abs(fr) > bw * margin / 2] = 0; y = np.fft.ifft(F).astype(np.complex64)
    from scipy.signal import welch
    fw, Pw = welch(y, fs=FS, nperseg=1024, return_onesided=False); o = np.argsort(fw); fw, Pw = fw[o], Pw[o]
    inb = np.abs(fw) <= bw * margin / 2; Pdb = 10 * np.log10(Pw + 1e-30)
    on = inb & (Pdb > Pdb[inb].max() - 15); f_lo, f_hi = fw[on].min(), fw[on].max()
    bw2 = max(f_hi - f_lo, 0.5e6); c2 = (f_lo + f_hi) / 2
    y = (y * np.exp(-2j * np.pi * c2 * np.arange(N) / FS)).astype(np.complex64)
    F = np.fft.fft(y); F[np.abs(fr) > bw2 * margin / 2] = 0
    return np.fft.ifft(F).astype(np.complex64), float(bw2)


def piece_stats(iq):
    """조각의 시간 칸(204 µs)별 전력에서 켜진 세기(90퍼센타일)·꺼진 바닥(10퍼센타일)과 신호가 있는 시간 범위를 구한다."""
    cell = int(FS * 0.1 / 490); p = (np.abs(iq[: len(iq) // cell * cell]) ** 2).reshape(-1, cell).mean(1)
    noise, burst = float(np.percentile(p, 10)), float(np.percentile(p, 90))
    on = np.where(p > noise * 4)[0]
    t0, t1 = (on[0] * cell / FS, (on[-1] + 1) * cell / FS) if len(on) else (0.0, WIN)
    return dict(noise=noise, burst=burst, t0=t0, t1=t1)


def tampere_piece(stem, t0, part="train"):
    """Tampere 2.4 GHz 녹음(120 MS/s, int16 번갈아)에서 t0 초부터 20 ms 의 넓은 링크 조각.
    실제 그림 분할(99 문서 6.3절)과 같은 시간 구간만 허용한다: part="train" 은 학습 구간 0~0.56초, part="test" 는 테스트 구간 0.80~1.00초.
    검증 구간 0.60~0.76초는 쓰지 않는다."""
    if part == "train": assert t0 >= 0 and t0 + WIN <= 0.56 + 1e-9, "Tampere 학습용 조각은 학습 구간(0~0.56초)에서만"
    elif part == "val": assert t0 >= 0.60 - 1e-9 and t0 + WIN <= 0.76 + 1e-9, "Tampere 검증용 조각은 검증 구간(0.60~0.76초)에서만(10-07 v7)"
    elif part == "test": assert t0 >= 0.80 - 1e-9 and t0 + WIN <= 1.0 + 1e-9, "Tampere 평가용 조각은 테스트 구간(0.80~1.00초)에서만"
    else: raise ValueError(part)
    fs_in = 120e6; n = int(fs_in * WIN)
    raw = np.fromfile(ROOT / f"data/raw/tampere/{stem}.bin", dtype=np.int16, count=2 * n, offset=4 * int(t0 * fs_in)).astype(np.float32)
    x = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    x60 = resample_poly(x, 1, 2).astype(np.complex64); L = find_wide_link(x60, 60e6)
    if L is None: return None
    fmid, bw, _, _ = L; iq, bw = cut_link(x, fs_in, fmid, bw)
    return dict(src=f"Tampere {stem} {t0:.2f}s", iq=iq, bw=bw, **piece_stats(iq))


def rfuav_piece(device, file_idx=0, t0=0.3):
    """RFUAV 녹음(100 MS/s complex64, 중심 2.44 GHz)에서 t0 초부터 20 ms 의 넓은 링크 조각."""
    import rfuav
    f = rfuav.iq_files(device)[file_idx]; x = rfuav.read_chunk(f, t0, WIN)
    x50 = resample_poly(x, 1, 2).astype(np.complex64); L = find_wide_link(x50, 50e6)         # 100 MHz 전체는 무거워 50 MHz 가운데서 찾음
    if L is None: return None
    fmid, bw, _, _ = L; iq, bw = cut_link(x, 100e6, fmid, bw)
    return dict(src=f"RFUAV {device} {Path(f).name} {t0:.2f}s", iq=iq, bw=bw, **piece_stats(iq))


def add_drone(x, piece, rng, snr_db, floor=1.0, fo=None, max_noise_ratio=0.3):
    """조각을 창 안 자리 fo(기본 무작위)에 더한다. snr_db = 신호가 켜진 순간의 1 Hz 당 세기가 바닥보다 얼마나 큰가.
    조각 안에 든 원래 녹음의 잡음이 우리 바닥의 max_noise_ratio 를 넘지 않게 세기를 깎는다(넘으면 띠 모양 얼룩이 생김).
    반환 (iq, 박스, 실제로 쓴 snr_db)."""
    bw = piece["bw"]; p = piece["iq"]
    lo, hi = -FS / 2 + bw / 2 + 0.5e6, FS / 2 - bw / 2 - 0.5e6
    if fo is None: fo = float(rng.uniform(lo, hi)) if hi > lo else 0.0        # 창보다 넓은 링크(DJI 20 MHz 영상)는 가운데에 두고 창에 잘린 채로 넣는다
    s2 = _rms_for(snr_db, bw, floor) ** 2 / piece["burst"]                    # 켜진 순간 세기를 목표에 맞추는 배율(전력)
    r = piece["noise"] * s2 * FS / (bw * floor)
    if r > max_noise_ratio: s2 *= max_noise_ratio / r
    snr_eff = 10 * np.log10(piece["burst"] * s2 / bw / (floor / FS))
    t = np.arange(N) / FS
    y = x + (p * np.sqrt(s2) * np.exp(2j * np.pi * fo * t)).astype(np.complex64)
    return y, ("drone", piece["t0"], piece["t1"], max(fo - bw / 2, -FS / 2), min(fo + bw / 2, FS / 2)), float(snr_eff)


def to_gray(x):
    """20 ms IQ → 640×640 그레이(uint8), 위가 높은 주파수."""
    S, _, _ = spectrogram_db(x, FS, nfft=NFFT, hop=HOP); return np.array(to_image(S))


DC_HALF = 100e3                                            # 10-07: 가운데 줄(DC) 지우기 폭 ±100 kHz (HackRF 는 평균을 빼도 0 Hz +10 dB, ±40 kHz +1 dB, ±100 kHz +0.4 dB 로 천천히 줄어듦)


def spectrogram_v7(x):
    """10-07 새 방법의 스펙트로그램(dB): 20 ms IQ 의 평균을 빼고 스펙트로그램을 만든 뒤, 0 Hz ± DC_HALF 줄들을
    같은 쪽 바로 바깥 줄(가운데에서 DC_HALF 만큼 더 떨어진 줄)의 값으로 같은 시각 그대로 채운다.
    학습 그림·테스트 그림·시연 프로그램 모두 이 함수로 그림을 만든다."""
    y = np.asarray(x, dtype=np.complex64); y = y - y.mean()
    S, f, _ = spectrogram_db(y, FS, nfft=NFFT, hop=HOP); k0 = int(np.argmin(np.abs(f))); K = int(np.ceil(DC_HALF / (FS / NFFT)))
    S = S.copy()
    for d in range(-K + 1, K):
        S[k0 + d] = S[k0 + (d + K if d >= 0 else d - K)]
    return S


def to_gray_v7(x):
    """10-07 새 방법의 그림: 가운데 줄을 지운 스펙트로그램 → 640×640 그레이."""
    return np.array(to_image(spectrogram_v7(x)))


def box_px(b):
    """박스(종류, t0, t1, f_lo, f_hi) → 픽셀 (x0, y0, x1, y1)."""
    _, t0, t1, fl, fh = b
    return t0 / WIN * PX, (0.5 - fh / FS) * PX, t1 / WIN * PX, (0.5 - fl / FS) * PX
