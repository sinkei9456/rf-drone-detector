"""링크 단위 자동 라벨 규칙.

입력: 파워 스펙트로그램 P[freq(오름차순), time] (선형). 출력: 링크 박스 목록.
1) 버스트 검출: 행별 잡음 바닥(시간 10퍼센타일) 대비 +9 dB 이상인 연결 영역.
2) 스퍼 제외(기본 꺼짐): AI Hub uav01 의 조종 링크가 0.1 s 내내 켜진 협대역 톤이라 "연속=스퍼" 규칙은 쓸 수 없다.
   스퍼는 무드론 배경 캡처에서 주파수 점유율로 따로 판별한다(추후).
3) 묶음:
   - 광대역(폭 ≥ WIDE_MHZ): 주파수 범위가 80 % 이상 겹치는 버스트들 = 한 링크(고정 채널). class 0 drone_ofdm.
   - 협대역: 폭이 비슷한(비율 4배 이내) 버스트 3개 이상 = 한 링크(홉핑). 겹치는 같은 클래스 박스는 합침. class 1 drone_fhss.
   박스 = 주파수: 링크가 차지한 범위(고정 채널은 채널 폭, 홉핑은 홉 범위), 시간: 첫 버스트 시작 ~ 마지막 버스트 끝.
클래스 이름은 CLASSES. Wi-Fi/BT(2, 3)는 합성기에서만 붙는다(실측 AI Hub 에는 없음).
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage

CLASSES = {0: "drone_wide", 1: "drone_narrow", 2: "wifi", 3: "bt"}   # wide: ≥4 MHz 버스트(OFDM 블록·광대역 펄스), narrow: 협대역 홉/버스트
WIDE_MHZ = 4.0
THR = 8.0                       # +9 dB
SPUR_TIME_FRAC = 1.1              # >1 이면 스퍼 규칙 비활성
SPUR_MAX_MHZ = 0.3
MIN_H_MHZ = 0.06
MIN_T_FRAMES = 2
WIDE_MIN_MS = 1.0
WIDE_MIN_FILL = 0.5           # 촘촘한 톤 다발(채움 낮음)은 광대역이 아님
NARROW_MIN_N = 3
NARROW_BW_RATIO = 4.0
POOL_F = 8                      # 8192 → 1024 행 (19.5 kHz/bin)


FLOOR_CLIP_DB = None             # 설정하면 행별 잡음 바닥을 전체 중앙값 ±이 값(dB) 안으로 제한 (DroneDetect·Tampere 용, 2026-09-30)


def detect_bursts(P: np.ndarray, df: float, dt: float, thr: float = THR, f_min: float = -10e6):
    """반환 bursts: (f0,f1,t0,t1) [Hz, s] 목록(스퍼 제외), spurs: 같은 형식."""
    H, W = P.shape
    Hp = H // POOL_F
    Pp = P[:Hp * POOL_F].reshape(Hp, POOL_F, W).max(axis=1)           # 주파수 방향 max-pool
    floor = np.percentile(Pp, 10, axis=1, keepdims=True) + 1e-20
    if FLOOR_CLIP_DB is not None:   # 늘 켜진 신호 행(바닥이 신호가 됨)과 필터 바깥 어두운 행(작은 누설도 +9 dB)을 막는다
        ref = np.median(floor); c = 10 ** (FLOOR_CLIP_DB / 10); floor = np.clip(floor, ref / c, ref * c)
    mask = (Pp / floor) > thr
    mask = ndimage.binary_opening(mask, np.ones((2, 1)))
    lab, n = ndimage.label(mask)                                        # 4-연결(대각선 불연결)
    bursts, spurs = [], []
    dfp = df * POOL_F
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        fill = float((lab[sl] == k).mean())                                # bbox 안에서 켜진 비율
        f0, f1 = f_min + sl[0].start * dfp, f_min + sl[0].stop * dfp
        t0, t1 = sl[1].start * dt, sl[1].stop * dt
        if (f1 - f0) < MIN_H_MHZ * 1e6 or (sl[1].stop - sl[1].start) < MIN_T_FRAMES: continue
        if (t1 - t0) >= SPUR_TIME_FRAC * W * dt and (f1 - f0) <= SPUR_MAX_MHZ * 1e6:
            spurs.append((f0, f1, t0, t1)); continue
        bursts.append((f0, f1, t0, t1, fill))
    return bursts, spurs


def _overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0.0, hi - lo) / max(min(a[1] - a[0], b[1] - b[0]), 1e-9)


def group_links(bursts, wide_mhz: float = WIDE_MHZ):
    """반환 links: dict(cls, f0, f1, t0, t1, n, bw, period)"""
    is_wide = lambda b: (b[1] - b[0]) >= wide_mhz * 1e6 and (b[3] - b[2]) >= WIDE_MIN_MS * 1e-3 and (len(b) < 5 or b[4] >= WIDE_MIN_FILL)
    wide = [b for b in bursts if is_wide(b)]
    narrow = [b for b in bursts if not is_wide(b)]
    links = []
    # 고정 채널: 주파수 범위 겹침으로 묶기
    used = [False] * len(wide)
    for i, b in enumerate(wide):
        if used[i]: continue
        grp = [b]; used[i] = True
        for j in range(i + 1, len(wide)):
            if not used[j] and _overlap(b[:2], wide[j][:2]) >= 0.8:
                grp.append(wide[j]); used[j] = True
        links.append(_mk(0, grp))
    # 광대역 링크의 주파수·시간 범위 안에 있는 협대역 조각은 블록 내부 질감 → 제거
    def inside(b):
        return any(b[0] >= L["f0"] - 0.1e6 and b[1] <= L["f1"] + 0.1e6 and b[2] >= L["t0"] and b[3] <= L["t1"] for L in links)
    narrow = [b for b in narrow if not inside(b)]
    # 홉핑: 폭이 비슷한 것끼리 (4배 이내), 시간순
    narrow.sort(key=lambda b: b[2])
    used = [False] * len(narrow)
    for i, b in enumerate(narrow):
        if used[i]: continue
        bw = b[1] - b[0]; grp = [b]; used[i] = True
        for j in range(i + 1, len(narrow)):
            bwj = narrow[j][1] - narrow[j][0]
            if not used[j] and 1 / NARROW_BW_RATIO <= bwj / bw <= NARROW_BW_RATIO:
                grp.append(narrow[j]); used[j] = True
        if len(grp) >= NARROW_MIN_N:
            links.append(_mk(1, grp))
    return merge_links(links)


def _iou(a, b):
    ix = max(0, min(a["t1"], b["t1"]) - max(a["t0"], b["t0"])); iy = max(0, min(a["f1"], b["f1"]) - max(a["f0"], b["f0"]))
    inter = ix * iy; ua = (a["t1"] - a["t0"]) * (a["f1"] - a["f0"]); ub = (b["t1"] - b["t0"]) * (b["f1"] - b["f0"])
    return inter / max(ua + ub - inter, 1e-12)


def merge_links(links, iou_thr: float = 0.3):
    """같은 클래스의 박스가 크게 겹치면 하나로."""
    links = list(links); changed = True
    while changed:
        changed = False
        for i in range(len(links)):
            for j in range(i + 1, len(links)):
                a, b = links[i], links[j]
                if a["cls"] == b["cls"] and _iou(a, b) > iou_thr:
                    links[i] = dict(cls=a["cls"], f0=min(a["f0"], b["f0"]), f1=max(a["f1"], b["f1"]), t0=min(a["t0"], b["t0"]), t1=max(a["t1"], b["t1"]),
                                    n=a["n"] + b["n"], bw=(a["bw"] * a["n"] + b["bw"] * b["n"]) / (a["n"] + b["n"]), period=a["period"])
                    links.pop(j); changed = True; break
            if changed: break
    return links


def _mk(cls, grp):
    f0 = np.median([g[0] for g in grp]) if cls == 0 else min(g[0] for g in grp)
    f1 = np.median([g[1] for g in grp]) if cls == 0 else max(g[1] for g in grp)
    t0, t1 = min(g[2] for g in grp), max(g[3] for g in grp)
    starts = sorted(g[2] for g in grp)
    period = float(np.median(np.diff(starts))) if len(starts) > 2 else float("nan")
    return dict(cls=cls, f0=f0, f1=f1, t0=t0, t1=t1, n=len(grp), bw=float(np.median([g[1] - g[0] for g in grp])), period=period)


def links_to_yolo(links, f_span: float, t_span: float, f_min: float = -10e6):
    """표시 좌표(x=시간 0..1, y=위가 고주파 0..1)의 YOLO 줄 목록 (cls xc yc w h)."""
    out = []
    for L in links:
        x0, x1 = L["t0"] / t_span, L["t1"] / t_span
        y0, y1 = 1 - (L["f1"] - f_min) / f_span, 1 - (L["f0"] - f_min) / f_span
        x0, x1, y0, y1 = [min(max(v, 0.0), 1.0) for v in (x0, x1, y0, y1)]
        if x1 - x0 < 0.003 or y1 - y0 < 0.003: continue
        out.append((L["cls"], (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
    return out


def label_power(P, df, dt, f_min=None):
    """f_min: 첫 행의 주파수. 기본 = 중심 0 기준(-창폭/2). 2026-09-30 이전에는 -10 MHz 고정이라 창 폭이 20 MHz 가 아니면
    (DroneDetect·Tampere 60 MHz, Mendeley 56 MHz) links_to_yolo(f_min=-창폭/2) 와 어긋나 박스가 (창폭/2 - 10 MHz) 만큼 위로 밀렸다."""
    if f_min is None: f_min = -P.shape[0] * df / 2
    b, s = detect_bursts(P, df, dt, f_min=f_min)
    return group_links(b), b, s
