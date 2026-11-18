"""HackRF 녹음 위치 (2026-10-09 폴더 정리 후). 스크립트들은 녹음 경로를 여기서만 가져온다 — 폴더를 또 옮기면 이 파일만 고친다.

data/raw/hackrf/
├── 1_nodrone_indoor/   드론 없음, 실내 (옛 이름: HackRF_2450MHz_20MSps_AntennaOff_5x1s_20261006, …_AntennaOn_NoDrone_5x1s_20261007)
├── 2_neo2_indoor/      Neo 2, 실내 (옛 이름: hackrf_neo2/Neo2_ON_OFF_20261008/OFF, ON)
└── 3_neo2_park/        Neo 2, 공원 (팀원 이름 그대로 00~13, 학습 제외 90·91 은 _excluded/)
조건 이름 "AntennaOff"·"AntennaOn_NoDrone" 은 예전 스크립트·결과 파일에 쓰인 그대로 둔다."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; RAW = ROOT / "data/raw/hackrf"
NODRONE = {"AntennaOff": RAW / "1_nodrone_indoor/20261006_antenna_off", "AntennaOn_NoDrone": RAW / "1_nodrone_indoor/20261007_antenna_on"}
NEO2_INDOOR = {"OFF": RAW / "2_neo2_indoor/20261008_drone_off", "ON": RAW / "2_neo2_indoor/20261008_drone_on"}
NEO2_PARK = RAW / "3_neo2_park"


def nodrone_files(cond, fid="*"):
    """드론 없는 실내 녹음(.cs8). cond = "AntennaOff" 또는 "AntennaOn_NoDrone", fid = "01"~"05" 또는 "*"."""
    return sorted((NODRONE[cond] / "recordings").glob(f"{fid}/*.cs8"))


def nodrone_file(cond, fid):
    f = nodrone_files(cond, fid); assert len(f) == 1, (cond, fid, f)
    return f[0]
