"""생성 간섭 모델 실험(원고 4–6장) 공통 정의 (2026-11-12).
fold(제품 계열 5개), 간섭 밀도 4조건, seed, 배경 잡음 파일, 드론 신호 세그먼트의 학습·검증 데이터·테스트 나누기, 저장 위치.
나누기 규칙(원고 표 2, 보충 자료 표 S3 과 같은 수):
  - 평가 fold의 기종 = 미학습 계열 평가의 드론 신호 세그먼트(그 기종의 것 전부)
  - 나머지 네 fold 에서 validation = Tampere 를 뺀 기존 validation IQ 파일(DRFF-R2·DroneDetect)의 드론 신호 세그먼트, train = 그 밖 전부
    (Tampere 는 기종당 IQ 파일이 하나라 모두 train, RFUAV 는 validation IQ 파일이 없음). 한 IQ 파일(group)은 한쪽에만 들어간다.
  - Neo 2 는 이 모음(runs/iq_pieces_v7)에 없다."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "data/derived/rev1"; RUNS = ROOT / "paper/runs/rev1"

FOLDS = [("f1_mini", "Mini", ["DJI Mini 5 Pro", "DJI Mini 4 Pro", "DJI Mini 3 Pro", "DJI Mini 3", "DJI Mavic Mini"]),
         ("f2_air", "Air", ["DJI Air 3S", "DJI Air 2S", "DJI Mavic Air 2", "DJI Mavic Air"]),
         ("f3_mavic", "Mavic", ["DJI Mavic 3", "DJI Mavic 3 Classic", "DJI Mavic Pro", "DJI Mavic Pro 2"]),
         ("f4_pro", "전문가용", ["DJI Phantom 4", "DJI Inspire 2", "DJI Matrice 100", "DJI Matrice 210"]),
         ("f5_other", "타 제조사", ["Parrot Disco", "Parrot Mambo", "Yuneec Typhoon H"])]
CONDS = [("c02", 0.02, "2 % 조건"), ("c20", 0.20, "20 % 조건"), ("c50", 0.50, "50 % 조건"), ("c80", 0.80, "80 % 조건")]   # 목표 = train 스펙트럼 점유율 중앙값
REPS = (1, 2)                                                   # seed 2개: 합성 난수와 학습 난수를 함께 바꾼다(원고 3.5절)
N_TRAIN, N_VAL, N_TEST_DRONE, N_TEST_NODRONE = 6000, 600, 500, 500
BG = {"train": "Park02_1", "val": "Park02_2", "test": "Park02_3"}   # 공원 배경 0–1 s / 1–2 s / 2–3 s
# 난수 씨앗의 첫 숫자(데이터셋 역할마다 다르게): (드론·배경, 와이파이, 블루투스)
SEED = {"train": (11, 13, 14), "val": (12, 15, 16), "test_drone": (21, 25, 26), "test_nodrone": (23, 27, 28)}
N_WIFI = [0] * 12 + [1] * 14 + [2] * 14 + [3] * 12 + [4] * 12 + [6] * 12 + [8] * 10 + [10] * 8 + [12] * 6   # v10 규칙(src/synth_iq_build.N_WIFI)
N_BT_MAX = 30                                                   # v10 규칙: 블루투스 0–30개


def pieces():
    return np.load(ROOT / "runs/iq_pieces_v7.npy", mmap_mode="r"), json.load(open(ROOT / "runs/iq_pieces_v7_meta.json"))


def model_of(m):
    import piece_tables as P
    return P.kind(m["group"])[0]


def split_pieces(fold_key, meta=None):
    """반환 dict(train=[…], val=[…], test=[…]) 드론 신호 세그먼트 번호(runs/iq_pieces_v7 의 순서)."""
    meta = meta or pieces()[1]; models = dict((k, L) for k, _, L in FOLDS)[fold_key]
    out = dict(train=[], val=[], test=[])
    for i, m in enumerate(meta):
        if model_of(m) in models: out["test"].append(i)
        elif m["part"] == "val" and not m["group"].startswith("Tampere"): out["val"].append(i)
        else: out["train"].append(i)
    return out


def set_name(fold_key, cond, rep): return f"{fold_key}_{cond}_s{rep}"
def test_name(fold_key, cond): return f"test1_1_{fold_key}_{cond}"
def nodrone_name(cond): return f"test_nodrone_{cond}"
