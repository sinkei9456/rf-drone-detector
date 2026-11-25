"""잡음 세기(주파수 구간별 하위 10 % 세기의 중앙값) 계산 (2026-11-24). 원고 4.3절, 보충 자료 S7.1절.
정의: IQ 파일의 겹치지 않는 20 ms 구간을 모두 occupancy.power(97 시간 칸 × 512 주파수 구간, IQ 값의 제곱 단위)로 바꾸고,
주파수 구간마다 모든 시간 칸에서 하위 10 % 값을 구한 뒤, 그 512개 값의 중앙값을 IQ 파일 하나의 잡음 세기로 한다.
대상: 공원 배경(Park02_1, 실제 배경 모델이 RMS 정규화에 쓰는 공원 배경 0–1 s), 객차 드론 전원 꺼짐 IQ 파일 16개,
수원역 드론·조종기 전원 꺼짐 IQ 파일 4개(paper/runs/rev1/test1_4/clips.csv 의 drone == False).
출력: paper/runs/rev1/exp2/noise_level.json (IQ 파일별 값, 장소별 범위, 공원 대비 dB)
사용: python paper/src/noise_level.py"""
import csv, json, math, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]                                  # contest/d4d_2026
sys.path.insert(0, str(HERE))
import occupancy as E

PARK = PROJ / "data/derived/hackrf_clean/Park02_1.npy"
CLIPS = PROJ / "paper/runs/rev1/test1_4/clips.csv"
OUT = PROJ / "paper/runs/rev1/exp2/noise_level.json"


def level(windows):
    P = np.stack([E.power(w) for w in windows])          # (구간 수, 97, 512)
    return float(np.median(np.percentile(P.reshape(-1, E.NF), 10, axis=0)))


def main():
    x = np.load(PARK)
    park = level([x[i:i + E.N] for i in range(0, len(x) - E.N + 1, E.N)])
    rows = [r for r in csv.DictReader(open(CLIPS, encoding="utf-8")) if r["drone"] == "False"]
    files = {r["clip"]: {"group": r["group"], "venue": r["venue"], "value": level(E.read_iq(PROJ / r["path"]))} for r in rows}
    out = {"park_Park02_1": park, "files": files, "venues": {}}
    for name, pre in (("객차", "T_off"), ("수원역", "S_off")):
        v = [f["value"] for f in files.values() if f["group"].startswith(pre)]
        d = [10 * math.log10(a / park) for a in v]
        out["venues"][name] = {"n_files": len(v), "min": min(v), "max": max(v), "db_vs_park_min": min(d), "db_vs_park_max": max(d)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"공원 {park:.1f}")
    for k, s in out["venues"].items():
        print(f"{k} IQ 파일 {s['n_files']}개 {s['min']:.1f}–{s['max']:.1f}, 공원 대비 {s['db_vs_park_min']:.2f}–{s['db_vs_park_max']:.2f} dB")


if __name__ == "__main__":
    main()
