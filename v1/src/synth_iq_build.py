"""IQ 단계 합성 그림 만들기. 방식은 src/synth_iq.py(05 문서 5c절).

조각 모음 v2 (10-04, 체인 14) — 드론 영상 링크 IQ 조각을 **학습용**과 **평가용**으로 녹음·시간 구간 단위로 나눠 모은다.
  학습용(part=train): 실제 그림 학습셋과 같은 녹음·구간에서만
      Tampere 2.4 GHz 녹음 0.02~0.54초(실제 분할의 학습 구간), DroneDetect CLEAN 녹음 00~02(실제 분할의 학습 녹음),
      RFUAV DJI Mini 4 Pro 영상 폭 10 MHz 녹음(pack1)
  평가용(part=test): 어떤 모델도 학습·검증에서 본 적 없는 녹음·구간에서만
      Tampere 2.4 GHz 녹음 0.80~1.00초(실제 분할의 테스트 구간), DroneDetect CLEAN 녹음 04(실제 분할의 테스트 녹음),
      RFUAV DJI Mini 3 전부(이 기종은 학습에 전혀 없음), RFUAV DJI Mini 4 Pro 영상 폭 20 MHz 녹음(pack2, 다른 녹음)
  조각 조건: 폭 3~18 MHz, 켜짐/꺼짐 세기 비 ≥ 15 dB, 켜진 시간 ≥ 2 ms.
  출력 runs/iq_pieces_v2.npy (complex64 [n, 400000]), runs/iq_pieces_v2_meta.json (조각마다 src, group, part, bw, 세기, 시간)

그림: 학습 N장·검증 M장은 학습용 조각만, 평가 K장은 평가용 조각만 쓴다. 바탕(잡음·와이파이·블루투스)·자리·세기는 그림마다 다른 난수
  (씨앗: 학습 1000+i, 검증 900000+i, 평가 5000000+i — 겹치지 않음).
  그림마다 와이파이 0~12개, 블루투스 0~30개. 드론은 학습·검증 그림의 70 %, 평가 그림의 60 %에만(1개 85 % / 2개 15 %), 세기 6~25 dB.
  라벨: 0 drone, 1 wifi, 2 bt (넣은 자리 그대로). 출력 <out>/{images,labels}/, train.txt, val.txt, test.txt, meta.json

(10-03 체인 13 의 옛 모음 runs/iq_pieces.npy 와 그림 data/derived/yolo_syn_iq/ 는 학습용·평가용 구분이 없던 것이라 체인 14 에서는 쓰지 않는다.)
사용: python src/synth_iq_build.py pieces ; python src/synth_iq_build.py build --n_train 4000 --n_val 400 --n_test 800 --out data/derived/yolo_syn_iq_v2

v3 (10-06, 체인 16, --ver v3): v2 와 같은 출처 + **DRFF-R2 2.4 GHz** 조각, **조각 폭 상한 20 MHz**(사용자 결정). 바탕(잡음·와이파이·블루투스)은 v2 와 같은 방식.
  DRFF 조각: 학습용 = DRFF 학습셋 녹음(기체 1·2호기, 1대 기종 c1·c2)에서만, 평가용 = DRFF 테스트셋 녹음(4호기·Mini 3 Pro 2호기·1대 기종 c6)에서만.
      DRFF 검증셋 녹음·드론 없는 배경·영상 신호가 없어 뺀 Mini 4 Pro 3호기는 쓰지 않는다. 드론 신호 자리는 사람이 확인한 자리(drff_prep.bands).
      녹음마다 0.1~1.3초의 7곳에서 잘라 조건을 통과한 앞의 2개까지(녹음 하나가 조각을 독차지하지 않게).
  그림 씨앗: 학습 20,000,000+i, 검증 29,000,000+i, 평가 35,000,000+i (v2 와도, 셋끼리도 겹치지 않음).
  출력 runs/iq_pieces_v3.npy·_meta.json, data/derived/yolo_syn_iq_v3/
  사용: python src/synth_iq_build.py pieces --ver v3 ; python src/synth_iq_build.py build --ver v3 --n_train 4000 --n_val 400 --n_test 800

v4 (10-06, 체인 17, --ver v4, 사용자 승인): 조각은 v3 그대로(runs/iq_pieces_v3.npy), **넣는 법 하나만 바꿈.**
  폭 18 MHz 가 넘는 조각은 그림(20 MHz) 위나 아래로 일부가 걸쳐 나가게 넣는다: 보이는 쪽 경계와 그림 끝 사이 2~10 MHz(무작위),
  보이는 높이 10~18 MHz. 그림 밖으로 나가는 부분은 넣기 전에 잘라 낸다(감겨 들어오지 않게). 18 MHz 이하 조각은 v3 와 같다.
  이유: v3 학습 그림의 drone 박스 41 %가 그림 높이 전체를 덮어, 모델이 잡음뿐인 빈 그림도 drone 으로 봄(99 문서 4c절).
  학습셋·검증셋만 만든다(테스트셋은 v3 평가 800장을 그대로 씀). 씨앗: 학습 40,000,000+i, 검증 49,000,000+i.
  사용: python src/synth_iq_build.py build --ver v4 --n_train 4000 --n_val 400 --n_test 0

v5 (10-06, 체인 18, --ver v5, 사용자 승인 "F 시작"): v4 + **빈 그림**. 이유: 학습 합성 4,000장 중 완전히 빈 그림이 5~12장뿐이라
  모델마다 빈 그림에서 "드론"이라고 하는지가 제멋대로였다(99 문서 4d절, runs/review/noise_probe_v8.txt).
  그림의 15 %는 박스 없는 빈 그림: 그중 1/3 은 고른 잡음, 2/3 은 위아래가 어두운 잡음(수신 필터가 좁은 꼴).
  나머지 그림의 20 %에도 같은 어두운 띠를 씌운다(IQ 전체에 걸어 실제 필터처럼). 어두운 띠 안으로 들어간 박스 부분은 잘라 내고, 0.5 MHz 보다 얇게 남으면 뺀다.
  어두운 띠: 위·아래 각각 폭 0.5~4 MHz(따로 뽑음), 어둡기 6~25 dB, 경계는 반은 칼같이·반은 0.5~3 MHz 에 걸쳐 완만하게.
  조각·넣는 법은 v4 와 같다. 학습셋·검증셋만 만든다(테스트셋은 v3 평가 800장). 씨앗: 학습 60,000,000+i, 검증 69,000,000+i.
  사용: python src/synth_iq_build.py build --ver v5 --n_train 4000 --n_val 400 --n_test 0

v6 (10-07, 체인 20, --ver v6, 사용자 결정 "① 실제 녹음을 바탕으로"): v5 와 같되 **바탕 잡음을 팀원의 HackRF 실제 녹음**으로 바꾼다.
  이유: 우리가 만든 고른 잡음으로 학습한 F1000 이 HackRF 빈 녹음에서 10장 묶음 25개 중 23~24개를 '드론'이라 함(runs/review/hackrf_look.txt).
  바탕 = data/raw/hackrf 의 두 조건(안테나 뺌 / 안테나 끼움·드론 없음, 2.45 GHz, 20 MS/s, int8, 이득 32 dB)에서 무작위로 고른 파일의
         무작위 시점 20 ms. 파일마다 잡음 세기(DC 를 뺀 분산)로 나눠 잡음 세기 1 로 맞춘다(와이파이·블루투스·드론의 세기 기준과 같게). DC 는 그대로 둔다.
  파일 나누기(셋끼리 겹치지 않음): 학습 그림 = 파일 01·02, 검증 그림 = 03, 테스트 그림 = 04·05 (두 조건 모두).
  빈 그림 15 %(박스 없음) = HackRF 녹음 20 ms 그대로. 위아래 어두운 띠는 넣지 않는다(이 이득에서 HackRF 그림에 없음).
  조각·넣는 법(넓은 조각 걸쳐 넣기)은 v5 와 같다. 테스트셋 800장도 만든다(평가용 조각 + 파일 04·05 바탕).
  씨앗: 학습 80,000,000+i, 검증 89,000,000+i, 테스트 95,000,000+i.
  사용: python src/synth_iq_build.py build --ver v6 --n_train 1000 --n_val 400 --n_test 800

v7 (10-07, 새 출발점, 사용자 결정 "위 계획대로 간다"): 라벨이 정확한 재료만 쓰는 합성.
  바탕 = HackRF **안테나 뺀 녹음**에서 블루투스로 보이는 덩어리를 지운 것(src/hackrf_clean.py → data/derived/hackrf_clean/AntennaOff_<번호>.npy).
         학습 그림 = 첫째·둘째 녹음, 검증 그림 = 셋째, 테스트 그림 = 넷째·다섯째. 파일마다 잡음 세기 1 로 맞춤. (안테나 끼운 녹음은 실측 테스트에만 씀)
  와이파이·블루투스 = 우리가 만든 것만(넣은 위치가 정답). 실제 녹음에서 자동 규칙으로 잘라 낸 와이파이·블루투스 조각은 쓰지 않는다.
  드론 = 드론 조각: v3 조각 모음의 학습용·테스트용 + **검증용 조각을 새로 자름**(Tampere 0.60~0.76초, DroneDetect 간섭 없는 녹음의 넷째(03),
         DRFF-R2 검증용 드론 녹음). 학습 그림은 학습용 조각만, 검증 그림은 검증용 조각만, 테스트 그림은 테스트용 조각만.
  빈 그림 15 %, 넓은 조각은 그림 위나 아래로 걸쳐 넣음(v4~v6 와 같음), 어두운 띠 없음.
  그림은 synth_iq.to_gray_v7(평균을 빼고 가운데 ±100 kHz 줄을 바로 바깥 줄로 채운 스펙트로그램)로 만든다 — 테스트 그림·시연 프로그램과 같게.
  씨앗: 학습 100,000,000+i, 검증 109,000,000+i, 테스트 115,000,000+i.
  사용: python src/synth_iq_build.py pieces_val ; python src/synth_iq_build.py build --ver v7 --n_train 6000 --n_val 600 --n_test 1000

v8 (10-07, 사용자 결정 "세로 20 MHz × 가로 20 ms 고정, 시연 간격 20 MHz, 학습·검증은 합성만"): v7 과 같은 재료(바탕·조각)에 **잘린 신호**를 넣는다.
  시연은 폭 20 MHz 를 고정하고 중심 주파수만 바꿔 가며 보므로, 신호가 그림 가장자리에서 잘려 보이는 경우가 흔하다.
  와이파이: 가운데 위치를 고르게 뽑되 폭의 20 % 이상 보이게(행사장에 흔한 잘린 와이파이를 와이파이로 배우게).
  블루투스: 0.5 MHz 이상 보이게(그림에서 세로 16 픽셀, 기존 블루투스 덩어리의 절반).
  드론: 30 %는 통째로, 나머지는 가운데가 그림 안 아무 데나 = 폭의 50 % 이상 보임(20 MHz 간격으로 돌면 어떤 드론이든 적어도 한 그림에 50 % 이상 보임).
        50 % 미만으로 보이는 드론은 넣지 않는다(라벨 없이 넣으면 "드론 = 아무것도 아님"으로 배움).
  잘린 신호도 1 Hz 당 세기는 통째일 때와 같게(v7 까지는 잘린 와이파이가 보이는 폭에 전체 세기가 몰려 더 세게 그려졌음).
  박스는 그림 안에 보이는 부분만. 씨앗: 학습 120,000,000+i, 검증 129,000,000+i, 테스트 135,000,000+i.
  사용: python src/synth_iq_build.py build --ver v8 --n_train 6000 --n_val 600 --n_test 1000

v9 (10-09 밤, 사용자 결정 — doc/report_261010.md 1·2장): v8 + **DJI Neo 2 조각** + **공원 바탕**.
  바탕: 그림마다 반반 — 실내 안테나 뺀 녹음(블루투스 지움, 학습 01·02 / 검증 03) 또는 공원 02(짧은 신호 지움, src/neo2_pieces.py,
        학습 Park02_1·2 / 검증 Park02_3). 잡음 세기 1 로 맞춤(v7·v8 과 같음).
  드론: 드론 칸마다 1/3 은 Neo 2 조각(runs/neo2_pieces.npy, 학습 = 공원 01·03·09, 검증 = 공원 10), 2/3 은 v8 과 같은 조각.
  Neo 2 넣는 법 = **드론 대역 갈아 끼우기**: 바탕의 [fo−5, fo+5] MHz 성분을 줄이고 조각(같은 HackRF·같은 이득이라 잡음 세기가 같음)을 넣는다.
        조각의 잡음(원래 위치의 공원 잡음 모양 Lp)과 바탕 잡음(Lb, 안테나 뺀 녹음은 모양이 다름)이 같아지도록, 바탕 성분은 sqrt(1 − a²·Lp/Lb) 배로 남긴다
        (a² = 조각 세기 배율, a² ≤ min(Lb/Lp) 라 남기는 배율이 0 아래로 내려가지 않음). 그래서 약한 조각(10 m 의 3–6 dB)도 얼룩 없이 들어간다.
        세기: 반은 녹음 그대로, 반은 "멀리 있는 드론 흉내"로 신호만 줄여 드론 대역 세기를 1 dB 와 원래 세기 사이로(사용자 제안, 1 dB 아래는 안 보여 정답이 틀어짐).
        위치: v8 과 같다(30 % 통째로, 나머지는 가운데가 그림 안 = 폭의 50 % 이상 보임). 박스 = 드론 채널 10 MHz 중 그림 안 부분 × 자동 레이블의 시간 범위.
        Neo 2 는 와이파이·블루투스보다 먼저 넣는다(나중에 넣으면 대역 안의 와이파이·블루투스가 줄어 라벨과 어긋남). 다른 드론 조각은 v8 처럼 나중에 더한다.
  학습셋·검증셋만 만든다(테스트 ③ 은 v8 테스트 1,000장 그대로). 씨앗: 학습 140,000,000+i, 검증 149,000,000+i.
  사용: python src/synth_iq_build.py build --ver v9 --n_train 6000 --n_val 600 --n_test 0"""
import sys, json, argparse
from pathlib import Path
from multiprocessing import Pool
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import sys as _sys; _sys.path.insert(0, str(ROOT / "src")); import hackrf_paths as HP   # 10-09 HackRF 폴더 정리: 녹음 경로는 hackrf_paths 에서
import synth_iq as S
from PIL import Image
VER = "v2"
PIECES, META = ROOT / "runs/iq_pieces_v2.npy", ROOT / "runs/iq_pieces_v2_meta.json"
OUT = ROOT / "data/derived/yolo_syn_iq_v2"
SEEDS = {"v2": (1000, 900000, 5000000), "v3": (20_000_000, 29_000_000, 35_000_000), "v4": (40_000_000, 49_000_000, 55_000_000), "v5": (60_000_000, 69_000_000, 75_000_000), "v6": (80_000_000, 89_000_000, 95_000_000), "v7": (100_000_000, 109_000_000, 115_000_000), "v8": (120_000_000, 129_000_000, 135_000_000), "v9": (140_000_000, 149_000_000, 155_000_000), "v10": (160_000_000, 169_000_000, 175_000_000), "v11": (180_000_000, 189_000_000, 195_000_000)}
WIDE = 18e6                                                     # v4: 이보다 넓은 조각은 그림 위나 아래로 걸쳐 넣음


def set_ver(v):
    global VER, PIECES, META, OUT
    VER = v; pv = "v3" if v in ("v4", "v5", "v6") else "v7" if v in ("v8", "v9", "v10") else v                    # v4~v6 은 v3 조각을 그대로 씀, v7 은 v3 + 검증용 조각(iq_pieces_v7), v11 은 v7 + 새 조각(iq_pieces_v11, src/pieces_v11.py)
    PIECES, META = ROOT / f"runs/iq_pieces_{pv}.npy", ROOT / f"runs/iq_pieces_{pv}_meta.json"; OUT = ROOT / f"data/derived/yolo_syn_iq_{v}"
    S.LINK_MAX = 20e6 if v in ("v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11") else 18e6


def ok(p):
    return p is not None and 3e6 <= p["bw"] <= S.LINK_MAX and p["burst"] / max(p["noise"], 1e-30) >= 10 ** 1.5 and (p["t1"] - p["t0"]) >= 2e-3


def drff_pieces(stem, part):
    """DRFF 녹음 하나에서 조건을 통과한 조각 앞의 2개까지(0.1~1.3초 7곳). 자리 = 사람이 확인한 드론 신호 자리."""
    import drff_prep as DP
    band = DP.bands()[stem]; x, fc = DP.load(DP.RAW / f"dataset6-wifi_mixed/{stem}.mat"); out = []
    for t0 in (0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3):
        seg = x[int(t0 * 1e8): int(t0 * 1e8) + 2_000_000]
        iq, bw = S.cut_link(seg, 100e6, (band[0] + band[1]) / 2 - fc, band[1] - band[0])
        p = dict(src=f"DRFF {stem} {t0:.1f}s", iq=iq, bw=bw, **S.piece_stats(iq))
        if ok(p): p["part"], p["group"] = part, f"DRFF {stem}"; out.append(p)
        if len(out) == 2: break
    return out


def dd_piece(f, t0):
    fs = 60e6; x = np.fromfile(f, dtype=np.complex64, count=int(fs * S.WIN), offset=8 * int(t0 * fs))
    if len(x) < int(fs * S.WIN): return None
    L = S.find_wide_link(x, fs)
    if L is None: return None
    iq, bw = S.cut_link(x, fs, L[0], L[1])
    return dict(src=f"DroneDetect {f.parts[-3]}/{f.parts[-2]}/{f.name} {t0:.1f}s", iq=iq, bw=bw, **S.piece_stats(iq))


def candidates():
    """(종류, ..., part, group). group = 같은 녹음·같은 시간 구간을 나타내는 이름(학습용과 평가용이 겹치면 안 됨)."""
    jobs = []
    for f in sorted((ROOT / "data/raw/tampere").glob("*_2G*.bin")):
        for t0 in np.arange(0.02, 0.5401, 0.04): jobs.append(("tampere", f.stem, round(float(t0), 2), "train", f"Tampere {f.stem} 0.00-0.56s"))
        for t0 in np.arange(0.80, 0.9801, 0.03): jobs.append(("tampere", f.stem, round(float(t0), 2), "test", f"Tampere {f.stem} 0.80-1.00s"))
    import rfuav
    for dev in ("DJI MINI4 PRO", "DJI MINI3"):
        for i, f in enumerate(rfuav.iq_files(dev)):
            part = "train" if (dev == "DJI MINI4 PRO" and "VTSBW=10" in f) else "test"
            rel = f.split("rfuav/")[1]
            for t0 in ((0.1, 0.5) if dev == "DJI MINI4 PRO" else (0.1, 0.3, 0.5, 0.7)): jobs.append(("rfuav", dev, i, t0, part, f"RFUAV {rel}"))
    for f in sorted((ROOT / "data/raw/dronedetect/DroneDetect_V2/CLEAN").glob("*/*.dat")):
        if f.stat().st_size < 800_000_000: continue
        rec = f.stem[-2:]; part = "train" if rec in ("00", "01", "02") else ("test" if rec == "04" else None)
        if part is None: continue                                                          # 03 = 실제 분할의 검증 녹음, 안 씀
        for t0 in (0.3, 0.9, 1.5): jobs.append(("dd", str(f), t0, part, f"DroneDetect CLEAN/{f.parts[-2]}/{f.name}"))
    if VER == "v3":
        import drff_prep as DP
        EX = set(DP.excluded())
        for f in sorted((DP.RAW / "dataset6-wifi_mixed").glob("*.mat")):
            part = DP.split_of(f.stem)
            if f.stem in EX or part not in ("train", "test"): continue                       # 검증셋 녹음·뺀 녹음은 안 씀
            jobs.append(("drff", f.stem, part, f"DRFF {f.stem}"))
    return jobs


def _piece_job(j):
    try:
        if j[0] == "drff": return drff_pieces(j[1], j[2])
        if j[0] == "tampere": p = S.tampere_piece(j[1], j[2], part=j[3])
        elif j[0] == "rfuav": p = S.rfuav_piece(j[1], j[2], j[3])
        else: p = dd_piece(Path(j[1]), j[2])
    except AssertionError: raise
    except Exception as e:
        print("실패", j[:4], repr(e)[:80], flush=True); return []
    if not ok(p): return []
    p["part"], p["group"] = j[-2], j[-1]
    return [p]


def pieces():
    jobs = candidates()
    from collections import Counter
    print("후보", len(jobs), dict(Counter(j[-2] for j in jobs)), flush=True)
    S.LINK_MAX = 20e6 if VER == "v3" else 18e6
    with Pool(8) as pool: res = pool.map(_piece_job, jobs, chunksize=1)
    res = [p for r in res for p in r]
    arr = np.lib.format.open_memmap(PIECES, mode="w+", dtype=np.complex64, shape=(len(res), S.N))
    meta = []
    for i, r in enumerate(res):
        arr[i] = r.pop("iq"); meta.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()})
    arr.flush(); del arr; json.dump(meta, open(META, "w"), ensure_ascii=False, indent=0)
    for part in ("train", "test"):
        m = [x for x in meta if x["part"] == part]
        print(f"{part}: 조각 {len(m)}개, 출처 {dict(Counter(x['src'].split()[0] for x in m))}, 녹음·구간 {len(set(x['group'] for x in m))}개, 폭 중앙값 {np.median([x['bw'] for x in m]) / 1e6:.1f} MHz")
    print("PIECES_DONE", flush=True)


def pieces_val():
    """v7: 검증용 드론 조각을 새로 잘라 v3 조각 모음(학습용·테스트용)에 붙여 runs/iq_pieces_v7.npy 로 저장.
    검증용 = Tampere 각 녹음의 0.60~0.76초, DroneDetect 간섭 없는 녹음의 넷째(_03), DRFF-R2 검증용 드론 녹음(drff_prep.split_of == 'val')."""
    import drff_prep as DP
    S.LINK_MAX = 20e6; jobs = []
    for f in sorted((ROOT / "data/raw/tampere").glob("*_2G*.bin")):
        for t0 in np.arange(0.60, 0.7401, 0.035): jobs.append(("tampere", f.stem, round(float(t0), 3), "val", f"Tampere {f.stem} 0.60-0.76s"))
    for f in sorted((ROOT / "data/raw/dronedetect/DroneDetect_V2/CLEAN").glob("*/*.dat")):
        if f.stat().st_size < 800_000_000 or f.stem[-2:] != "03": continue
        for t0 in (0.3, 0.9, 1.5): jobs.append(("dd", str(f), t0, "val", f"DroneDetect CLEAN/{f.parts[-2]}/{f.name}"))
    EX = set(DP.excluded())
    for f in sorted((DP.RAW / "dataset6-wifi_mixed").glob("*.mat")):
        if f.stem not in EX and DP.split_of(f.stem) == "val": jobs.append(("drff", f.stem, "val", f"DRFF {f.stem}"))
    from collections import Counter
    print("검증용 후보", len(jobs), dict(Counter(j[0] for j in jobs)), flush=True)
    with Pool(8) as pool: res = pool.map(_piece_job, jobs, chunksize=1)
    res = [p for r in res for p in r]
    old = np.load(ROOT / "runs/iq_pieces_v3.npy", mmap_mode="r"); om = json.load(open(ROOT / "runs/iq_pieces_v3_meta.json"))
    arr = np.lib.format.open_memmap(ROOT / "runs/iq_pieces_v7.npy", mode="w+", dtype=np.complex64, shape=(len(om) + len(res), S.N))
    arr[:len(om)] = old; meta = list(om)
    for i, r in enumerate(res):
        arr[len(om) + i] = r.pop("iq"); meta.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in r.items()})
    arr.flush(); del arr; json.dump(meta, open(ROOT / "runs/iq_pieces_v7_meta.json", "w"), ensure_ascii=False, indent=0)
    m = [x for x in meta if x["part"] == "val"]
    print(f"val: 조각 {len(m)}개, 출처 {dict(Counter(x['src'].split()[0] for x in m))}, 녹음·구간 {len(set(x['group'] for x in m))}개, 폭 중앙값 {np.median([x['bw'] for x in m]) / 1e6:.1f} MHz")
    print("PIECES_VAL_DONE", flush=True)


# ---------------- 그림 ----------------
N_WIFI = ([0] * 12 + [1] * 14 + [2] * 14 + [3] * 12 + [4] * 12 + [6] * 12 + [8] * 10 + [10] * 8 + [12] * 6)
_W = {}


HK_FILES = {"train": ("01", "02"), "val": ("03",), "test": ("04", "05")}         # v6: HackRF 파일 나누기
HK_COND = ("AntennaOff", "AntennaOn_NoDrone")


def _hackrf():
    """v6: HackRF 파일마다 (int8 메모리 지도, 잡음 표준편차). 잡음 표준편차 = DC(평균)를 뺀 복소 표본의 제곱 평균의 제곱근."""
    if "hk" not in _W:
        hk = {}
        for c in HK_COND:
            for f in HP.nodrone_files(c):
                r = np.memmap(f, dtype=np.int8, mode="r"); x = r.astype(np.float32); z = x[0::2] + 1j * x[1::2]
                hk[(c, f.parent.name)] = (r, float(np.sqrt(np.mean(np.abs(z - z.mean()) ** 2)))); del x, z
        assert len(hk) == 10, f"HackRF 파일 {len(hk)}개"
        _W["hk"] = hk
    return _W["hk"]


def hackrf_cut(rng, split):
    """v6: 셋(split)에 정해진 파일 중 하나의 무작위 시점 20 ms → 잡음 세기 1 로 맞춘 복소 IQ. 반환 (iq, 출처)."""
    hk = _hackrf(); c = HK_COND[int(rng.integers(2))]; fid = HK_FILES[split][int(rng.integers(len(HK_FILES[split])))]
    r, sd = hk[(c, fid)]; n = len(r) // 2; off = int(rng.integers(0, n - S.N + 1))
    seg = np.asarray(r[2 * off: 2 * (off + S.N)], dtype=np.float32)
    return ((seg[0::2] + 1j * seg[1::2]) / sd).astype(np.complex64), dict(cond=c, file=fid, offset=off)


HK7_FILES = {"train": ("01", "02"), "val": ("03",), "test": ("04", "05")}       # v7: 안테나 뺀 녹음만(블루투스 지운 것)


def hackrf_cut_v7(rng, split):
    """v7: 셋에 정해진 '안테나 뺀 녹음(블루투스 지움)' 중 하나의 무작위 시점 20 ms → 잡음 세기 1 로 맞춘 복소 IQ. 반환 (iq, 출처)."""
    if "hk7" not in _W:
        _W["hk7"] = {}
        for fid in ("01", "02", "03", "04", "05"):
            a = np.load(ROOT / f"data/derived/hackrf_clean/AntennaOff_{fid}.npy", mmap_mode="r")
            _W["hk7"][fid] = (a, float(np.sqrt(np.mean(np.abs(np.asarray(a) - np.asarray(a).mean()) ** 2))))
    fid = HK7_FILES[split][int(rng.integers(len(HK7_FILES[split])))]; a, sd = _W["hk7"][fid]
    off = int(rng.integers(0, len(a) - S.N + 1))
    return (np.asarray(a[off: off + S.N]) / sd).astype(np.complex64), dict(cond="AntennaOff_cleaned", file=fid, offset=off)


def _bank():
    if "arr" not in _W:
        _W["arr"] = np.load(PIECES, mmap_mode="r"); _W["meta"] = json.load(open(META))
        _W["pool"] = {p: [j for j, m in enumerate(_W["meta"]) if m["part"] == p] for p in ("train", "val", "test")}
    return _W["arr"], _W["meta"], _W["pool"]


def _img_job(a):
    split, i, seed, out = a; out = Path(out); part = split if VER in ("v7", "v8") else ("test" if split == "test" else "train")   # v7·v8: 검증 그림은 검증용 조각
    arr, meta, pool = _bank(); rng = np.random.default_rng(seed)
    empty = VER in ("v5", "v6", "v7", "v8") and rng.random() < 0.15   # v5~v8: 박스 없는 빈 그림
    base, hk = hackrf_cut(rng, split) if VER == "v6" else hackrf_cut_v7(rng, split) if VER in ("v7", "v8") else (None, None)   # v6~v8: 바탕 = HackRF 실제 녹음
    bgvis = []
    if VER == "v8":                                                    # v8: 잘린 와이파이·블루투스까지
        x, boxes, bgvis = S.make_background_v8(rng, 0, 0, base=base) if empty else S.make_background_v8(rng, int(rng.choice(N_WIFI)), int(rng.integers(0, 31)), base=base)
    else:
        x, boxes = S.make_background(rng, 0, 0, base=base) if empty else S.make_background(rng, int(rng.choice(N_WIFI)), int(rng.integers(0, 31)), base=base)
    used = []
    if not empty and rng.random() < (0.6 if split == "test" else 0.7):
        k = 2 if rng.random() < 0.15 else 1
        for j in rng.choice(pool[part], size=k, replace=False):
            p = dict(meta[j]); p["iq"] = np.array(arr[j]); snr_t = float(rng.uniform(6, 25)); fo, edge = None, None
            if VER == "v8":                                                # v8: 드론은 폭의 50 % 이상 보이게(30 %는 통째로)
                fo = S.place_drone_v8(rng, p["bw"]); p["iq"] = S.cut_to_window(p["iq"], fo)
                edge = None if abs(fo) + p["bw"] / 2 <= S.FS / 2 else ("top" if fo > 0 else "bottom")
            elif VER in ("v4", "v5", "v6", "v7") and p["bw"] > WIDE:
                p["iq"], fo, edge = straddle(p["iq"], p["bw"], rng)
            x, b, snr = S.add_drone(x, p, rng, snr_t, fo=fo); boxes.append(b)
            used.append(dict(piece=int(j), part=meta[j]["part"], group=meta[j]["group"], snr=round(snr, 1), edge=edge,
                             visible_mhz=round((min(b[4], S.FS / 2) - max(b[3], -S.FS / 2)) / 1e6, 2), bw_mhz=round(p["bw"] / 1e6, 2)))
    roll = None
    if VER == "v5" and (rng.random() < (2 / 3 if empty else 0.2)):
        x, roll = rolloff(x, rng)
        boxes = [(b[0], b[1], b[2], max(b[3], roll["lo"]), min(b[4], roll["hi"])) for b in boxes]
        boxes = [b for b in boxes if b[4] - b[3] >= 0.5e6]
    g = S.to_gray_v7(x) if VER in ("v7", "v8") else S.to_gray(x); stem = f"{split}_{i:05d}"
    Image.fromarray(g).save(out / "images" / f"{stem}.png")
    cls = {"drone": 0, "wifi": 1, "bt": 2}; lines = []
    for b in boxes:
        x0, y0, x1, y1 = S.box_px(b); x0, x1 = max(0, x0) / 640, min(640, x1) / 640; y0, y1 = max(0, y0) / 640, min(640, y1) / 640
        if x1 - x0 < 0.003 or y1 - y0 < 0.003: continue
        lines.append(f"{cls[b[0]]} {(x0+x1)/2:.5f} {(y0+y1)/2:.5f} {x1-x0:.5f} {y1-y0:.5f}")
    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return dict(stem=stem, split=split, seed=seed, n_drone=sum(b[0] == "drone" for b in boxes), n_wifi=sum(b[0] == "wifi" for b in boxes), n_bt=sum(b[0] == "bt" for b in boxes),
                pieces=used, empty=bool(empty), rolloff=roll, hackrf=hk, bg_visible=bgvis)


# ---------------- v9: Neo 2 조각 + 공원 바탕 ----------------
HK9 = {"train": [("off", "01"), ("off", "02"), ("park", "1"), ("park", "2")], "val": [("off", "03"), ("park", "3")]}
NEO2_P, NEO2_BW, NEO2_MIN_DB, NEO2_MIN_POOL_DB = 1 / 3, 10e6, 1.0, 1.5
# v10 (10-09 밤, 사용자 승인 — doc/report_261010.md): 약하게 만드는 조각의 절반은 1–2 dB(고도 30 m 수준), 비행 조각(01 빼고)의 절반은
#   켜질 때마다 1.5–3.5 ms 만 남기고 나머지를 공원 02 잡음(바탕과 같은 나누기: 학습 = Park02_1·2, 검증 = Park02_3)으로 바꿈("켜져 있는 시간 줄이기").
NEO2_WEAK_LOW_P, NEO2_WEAK_LOW_DB, NEO2_CUT_P, NEO2_CUT_MS = 0.5, 2.0, 0.5, (1.5, 3.5)
# v11 (10-10, 사용자 승인 — doc/extra/report_extra_v2.md): v10 과 모든 규칙이 같고, 드론 조각 모음만 v11(= v7 + 5.8 GHz·KU Leuven 새 조각, src/pieces_v11.py).
#   새 조각도 기존 조각처럼 조각 수에 비례해 뽑힌다. 1단계 테스트 부분에서 자른 조각은 없다.


def hackrf_cut_v9(rng, split):
    """v9: 바탕 반반(안테나 뺀 녹음 / 공원 02), 셋에 정해진 파일의 무작위 시점 20 ms → 잡음 세기 1. 반환 (iq, 출처)."""
    if "hk9" not in _W:
        _W["hk9"] = {}
        for src, k in HK9["train"] + HK9["val"]:
            a = np.load(ROOT / f"data/derived/hackrf_clean/{('AntennaOff_' if src == 'off' else 'Park02_') + k}.npy", mmap_mode="r"); v = np.asarray(a)
            _W["hk9"][(src, k)] = (a, float(np.sqrt(np.mean(np.abs(v - v.mean()) ** 2))))
    src = "park" if rng.random() < 0.5 else "off"; cand = [c for c in HK9[split] if c[0] == src]; key = cand[int(rng.integers(len(cand)))]
    a, sd = _W["hk9"][key]; off = int(rng.integers(0, len(a) - S.N + 1))
    return (np.asarray(a[off: off + S.N]) / sd).astype(np.complex64), dict(cond=key[0], file=key[1], offset=off)


def _neo2():
    if "neo2" not in _W:
        Z = np.load(ROOT / "runs/noise_shapes.npz"); f1 = Z["f_mhz"] * 1e6; ok = np.abs(f1) >= 0.25e6        # 0 Hz 줄 근처는 빼고 이웃으로 메움
        fr = np.fft.fftfreq(S.N, 1 / S.FS); meta = json.load(open(ROOT / "runs/neo2_pieces_meta.json"))
        _W["neo2"] = dict(arr=np.load(ROOT / "runs/neo2_pieces.npy", mmap_mode="r"), meta=meta, fr=fr,
                          pool={p: [j for j, m in enumerate(meta) if m["part"] == p and m["s0_db"] >= NEO2_MIN_POOL_DB] for p in ("train", "val")},   # 원래 세기 1.5 dB 미만 조각은 안 씀(넣은 뒤 1 dB 아래가 될 수 있음)
                          shape={k: np.interp(fr, f1[ok], Z[k][ok]) for k in ("off", "park")}, park=(f1[ok], Z["park"][ok]))
    return _W["neo2"]


def neo2_insert(x, j, rng, bg):
    """v9: Neo 2 조각 j 를 드론 대역 갈아 끼우기로 x(잡음 세기 1 인 바탕)에 넣는다. bg = 바탕 종류("off"/"park"). 반환 (iq, 박스, 정보)."""
    B = _neo2(); fr = B["fr"]; m = B["meta"][j]; t = np.arange(S.N) / S.FS
    fo = S.place_drone_v8(rng, NEO2_BW)
    p0, t1, on_ms = np.asarray(B["arr"][j]), m["t1"], None
    if VER in ("v10", "v11") and not m["session"].startswith("01") and rng.random() < NEO2_CUT_P:      # 켜져 있는 시간 줄이기(옮기기 전에)
        on_ms = float(rng.uniform(*NEO2_CUT_MS)); p0, t1 = neo2_shorten(p0, m, on_ms, rng)
    p = S.cut_to_window(p0, fo); P = np.fft.fft((p * np.exp(2j * np.pi * fo * t)).astype(np.complex64))
    M = (np.abs(fr - fo) <= NEO2_BW / 2) & (np.abs(fr) <= S.FS / 2 - 0.05e6)
    Lb = B["shape"][bg]; Lp = np.interp(fr - fo, *B["park"]); ratio = Lb[M] / Lp[M]; a2max, rmean = float(ratio.min()), float(ratio.mean())
    x0 = 10 ** (m["s0_db"] / 10) - 1; keep_db = 10 * np.log10(1 + x0 * a2max / rmean)
    if rng.random() < 0.5 or keep_db <= NEO2_MIN_DB: a2, how = a2max, "as recorded"
    else:
        hi = min(NEO2_WEAK_LOW_DB, keep_db) if VER in ("v10", "v11") and rng.random() < NEO2_WEAK_LOW_P else keep_db
        t_db = float(rng.uniform(NEO2_MIN_DB, hi)); a2, how = min((10 ** (t_db / 10) - 1) / x0 * rmean, a2max), "weakened"
    c = np.ones(S.N); c[M] = np.sqrt(np.clip(1 - a2 * Lp[M] / Lb[M], 0, None))
    y = np.fft.ifft(np.fft.fft(x) * c + np.sqrt(a2) * P).astype(np.complex64)
    b = ("drone", m["t0"], t1, max(fo - NEO2_BW / 2, -S.FS / 2), min(fo + NEO2_BW / 2, S.FS / 2))
    return y, b, dict(neo2=int(j), session=m["session"][:2], how=how, band_db=round(10 * np.log10(1 + x0 * a2 / rmean), 2), fo_mhz=round(fo / 1e6, 2),
                      visible_mhz=round((b[4] - b[3]) / 1e6, 2), bw_mhz=10.0, on_ms=None if on_ms is None else round(on_ms, 2), t1_cut=round(m["t1"] - t1, 5))


def neo2_bursts(x):
    """v10: 조각(드론 채널만 남긴 IQ)에서 신호가 켜진 시간들 [(시작, 끝) 샘플]. 세기를 51 µs 씩 평균해 아래 10 % 값(잡음)과 위 10 % 값(신호)의 가운데를 넘는 시간,
    0.2 ms 보다 짧은 틈은 잇고 0.5 ms 보다 짧은 덩어리는 버림. 신호와 잡음 차이가 잡음의 20 % 보다 작으면 [] (자르지 않음)."""
    e = np.convolve(np.abs(x) ** 2, np.ones(1024) / 1024, "same"); lo, hi = np.quantile(e, 0.1), np.quantile(e, 0.9)
    if hi - lo < 0.2 * lo: return []
    on = e > (lo + hi) / 2; d = np.diff(np.r_[0, on.astype(int), 0]); out = []
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if out and a - out[-1][1] < 0.2e-3 * S.FS: out[-1][1] = b
        else: out.append([a, b])
    return [(a, b) for a, b in out if b - a >= 0.5e-3 * S.FS]


def neo2_noise(part, rng):
    """v10: 공원 02 잡음 20 ms(지운 녹음, 학습 = Park02_1·2, 검증 = Park02_3)를 조각과 같은 방법으로(드론 채널 ±5 MHz 만, 0 Hz ±100 kHz 는 바깥 값으로) 만든 것. 잡음 세기 1 기준."""
    keys = [k for c, k in HK9[part] if c == "park"]; a, sd = _W["hk9"][("park", keys[int(rng.integers(len(keys)))])]; off = int(rng.integers(0, len(a) - S.N + 1))
    z = (np.asarray(a[off: off + S.N]) / sd).astype(np.complex64); z = z - z.mean(); fr = _neo2()["fr"]; F = np.fft.fft(z); F[np.abs(fr) > NEO2_BW / 2] = 0
    K = int(round(100e3 / (S.FS / S.N))); F[np.arange(K)] = F[np.arange(K) + K]; F[np.arange(S.N - K, S.N)] = F[np.arange(S.N - K, S.N) - K]
    return np.fft.ifft(F).astype(np.complex64)


def neo2_shorten(p, m, on_ms, rng, ramp_us=20):
    """v10: 켜진 덩어리마다 앞 on_ms 만 남기고 나머지를 공원 02 잡음으로 바꿈(경계 20 µs 섞음). 반환 (조각, 새 박스 끝 t1 초).
    정답 박스 끝: 원래 끝 t1 이 잘라낸 시간 안(또는 그 뒤 0.3 ms 안)에 있으면 잘라낸 시작으로 당김."""
    q = neo2_noise(m["part"], rng); w = np.ones(S.N); r = int(ramp_us * 1e-6 * S.FS); k = int(on_ms * 1e-3 * S.FS); t1 = m["t1"]
    for a, b in neo2_bursts(p):
        if b - a <= k: continue
        c = a + k; w[c:b] = 0; w[c - r:c] = np.minimum(w[c - r:c], 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, r)))
        if c / S.FS <= t1 <= b / S.FS + 0.3e-3: t1 = c / S.FS
    return (p * w + q * (1 - w)).astype(np.complex64), t1


def _img_job_v9(a):
    split, i, seed, out = a; out = Path(out); arr, meta, pool = _bank(); rng = np.random.default_rng(seed)
    empty = rng.random() < 0.15
    base, hk = hackrf_cut_v9(rng, split)
    drones = []
    if not empty and rng.random() < 0.7:
        drones = ["neo2" if rng.random() < NEO2_P else "other" for _ in range(2 if rng.random() < 0.15 else 1)]
    used, dboxes = [], []
    for d in drones:                                                   # Neo 2 먼저(갈아 끼우기), 와이파이·블루투스보다 앞
        if d == "neo2":
            j = int(rng.choice(_neo2()["pool"][split])); base, b, info = neo2_insert(base, j, rng, hk["cond"]); dboxes.append(b); used.append(info)
    x, boxes, bgvis = S.make_background_v8(rng, 0, 0, base=base) if empty else S.make_background_v8(rng, int(rng.choice(N_WIFI)), int(rng.integers(0, 31)), base=base)
    taken = set()
    for d in drones:
        if d != "other": continue
        j = int(rng.choice([q for q in pool[split] if q not in taken])); taken.add(j)
        p = dict(meta[j]); p["iq"] = np.array(arr[j]); snr_t = float(rng.uniform(6, 25))
        fo = S.place_drone_v8(rng, p["bw"]); p["iq"] = S.cut_to_window(p["iq"], fo)
        edge = None if abs(fo) + p["bw"] / 2 <= S.FS / 2 else ("top" if fo > 0 else "bottom")
        x, b, snr = S.add_drone(x, p, rng, snr_t, fo=fo); dboxes.append(b)
        used.append(dict(piece=j, part=meta[j]["part"], group=meta[j]["group"], snr=round(snr, 1), edge=edge,
                         visible_mhz=round((min(b[4], S.FS / 2) - max(b[3], -S.FS / 2)) / 1e6, 2), bw_mhz=round(p["bw"] / 1e6, 2)))
    boxes = boxes + dboxes
    g = S.to_gray_v7(x); stem = f"{split}_{i:05d}"; Image.fromarray(g).save(out / "images" / f"{stem}.png")
    cls = {"drone": 0, "wifi": 1, "bt": 2}; lines = []
    for b in boxes:
        x0, y0, x1, y1 = S.box_px(b); x0, x1 = max(0, x0) / 640, min(640, x1) / 640; y0, y1 = max(0, y0) / 640, min(640, y1) / 640
        if x1 - x0 < 0.003 or y1 - y0 < 0.003: continue
        lines.append(f"{cls[b[0]]} {(x0+x1)/2:.5f} {(y0+y1)/2:.5f} {x1-x0:.5f} {y1-y0:.5f}")
    (out / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return dict(stem=stem, split=split, seed=seed, n_drone=sum(b[0] == "drone" for b in boxes), n_wifi=sum(b[0] == "wifi" for b in boxes), n_bt=sum(b[0] == "bt" for b in boxes),
                n_neo2=sum(1 for u in used if "neo2" in u), pieces=used, empty=bool(empty), rolloff=None, hackrf=hk, bg_visible=bgvis)


def rolloff(x, rng):
    """v5: 위아래가 어두운 수신 필터 흉내. 위 dt·아래 db 폭(0.5~4 MHz)을 att dB(6~25) 어둡게, 경계는 반은 칼같이·반은 tw(0.5~3 MHz)에 걸쳐 완만하게.
    반환 (iq, dict(lo, hi = 밝은 띠의 아래·위 경계 Hz, ...))."""
    dt, db = rng.uniform(0.5e6, 4e6, 2); att = float(rng.uniform(6, 25)); tw = float(rng.uniform(0.5e6, 3e6)) if rng.random() < 0.5 else 0.0
    hi, lo = S.FS / 2 - dt, -S.FS / 2 + db; fr = np.fft.fftfreq(len(x), 1 / S.FS)
    over = np.maximum(fr - hi, 0) + np.maximum(lo - fr, 0)                     # 밝은 띠 밖으로 나간 거리
    gdb = -att * (np.minimum(over / tw, 1.0) if tw > 0 else (over > 0).astype(float))
    y = np.fft.ifft(np.fft.fft(x) * 10 ** (gdb / 20)).astype(np.complex64)
    return y, dict(lo=float(lo), hi=float(hi), att_db=round(att, 1), soft_mhz=round(tw / 1e6, 2))


def straddle(iq, bw, rng):
    """v4: 넓은 조각을 그림 위(top)나 아래(bottom)로 걸쳐 넣을 자리 fo 를 정하고, 그림 밖으로 나갈 부분은 미리 잘라 낸다.
    조각은 0 Hz 가운데에 [-bw/2, bw/2] 를 차지. 보이는 쪽 경계와 그림 끝 사이 m = 2~10 MHz."""
    m = float(rng.uniform(2e6, 10e6)); top = bool(rng.random() < 0.5)
    fo = (-S.FS / 2 + m + bw / 2) if top else (S.FS / 2 - m - bw / 2)          # top: 아래 경계가 그림 안, 위는 밖
    fr = np.fft.fftfreq(len(iq), 1 / S.FS); F = np.fft.fft(iq)
    keep = (fr + fo <= S.FS / 2 - 0.05e6) & (fr + fo >= -S.FS / 2 + 0.05e6)   # 옮긴 뒤 그림 안에 드는 성분만
    F[~keep] = 0
    return np.fft.ifft(F).astype(np.complex64), float(fo), ("top" if top else "bottom")


def build(n_train, n_val, n_test, workers, out):
    out = Path(out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    o = str(out)
    st, sv, se = SEEDS[VER]
    jobs = [("train", i, st + i, o) for i in range(n_train)] + [("val", i, sv + i, o) for i in range(n_val)] + [("test", i, se + i, o) for i in range(n_test)]
    if VER == "v6": _hackrf()                                           # 잡음 세기를 한 번만 재고 일꾼들이 물려받음
    if VER in ("v7", "v8"): hackrf_cut_v7(np.random.default_rng(0), "train")
    if VER in ("v9", "v10", "v11"): hackrf_cut_v9(np.random.default_rng(0), "train"); _neo2()
    with Pool(workers) as pool: rows = pool.map(_img_job_v9 if VER in ("v9", "v10", "v11") else _img_job, jobs, chunksize=8)
    json.dump(rows, open(out / "meta.json", "w"), ensure_ascii=False)
    for sp in ("train", "val", "test"):
        (out / f"{sp}.txt").write_text("\n".join(str((out / "images" / f"{r['stem']}.png").resolve()) for r in rows if r["split"] == sp) + "\n")
    import pandas as pd
    d = pd.DataFrame(rows)
    print(d.groupby("split")[["n_drone", "n_wifi", "n_bt"]].agg(["mean", "max"]).round(2))
    print("드론 있는 그림 비율", d.groupby("split").apply(lambda g: round(float((g.n_drone > 0).mean()), 3)).to_dict())
    if "n_neo2" in d: print("Neo 2 조각이 든 그림 비율", d.groupby("split").apply(lambda g: round(float((g.n_neo2 > 0).mean()), 3)).to_dict())
    print("장수", d.split.value_counts().to_dict())
    print("BUILD_DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["pieces", "pieces_val", "build"]); ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_val", type=int, default=400); ap.add_argument("--n_test", type=int, default=800); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None); ap.add_argument("--ver", default="v2", choices=["v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"]); a = ap.parse_args()
    set_ver(a.ver)
    if a.mode == "pieces": pieces()
    elif a.mode == "pieces_val": pieces_val()
    else: build(a.n_train, a.n_val, a.n_test, a.workers, a.out or str(OUT))
