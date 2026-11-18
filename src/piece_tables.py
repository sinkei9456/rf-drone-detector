"""보고서 공통 표: 합성 그림에 넣은 드론 조각의 출처·기종 (2026-10-10).
조각 수는 runs/iq_pieces_v7_meta.json(DRFF-R2·Tampere·DroneDetect·RFUAV 조각, 모델 1·2·3·1a·2a·3a 공통)과
runs/neo2_pieces_meta.json(공원 Neo 2 조각, 모델 1a·2a·3a 만, 드론 대역 세기 1.5 dB 이상만 씀)에서 센다.
section(neo2, level) → md 문자열. neo2=False 면 Neo 2 줄을 빼고(모델 1·2·3 만 다루는 보고서), level 은 표 제목의 # 수.
insert_before(doc, anchor, text) → anchor 줄 앞에 넣은 doc (anchor 가 정확히 한 번 있어야 함)."""
import json, re, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DRFF = {"mini5PRO": "DJI Mini 5 Pro", "mini4PRO": "DJI Mini 4 Pro", "mini3pro": "DJI Mini 3 Pro", "mavic3S": "DJI Air 3S", "mavicAir2s": "DJI Air 2S",
        "mavicAir2": "DJI Mavic Air 2", "mavic3C": "DJI Mavic 3 Classic", "mavic3": "DJI Mavic 3"}
TAMP = {"DJI_inspire_2": "DJI Inspire 2", "DJI_matrice_100": "DJI Matrice 100", "DJI_matrice_210": "DJI Matrice 210", "DJI_mavic_mini": "DJI Mavic Mini",
        "DJI_mavic_pro": "DJI Mavic Pro", "DJI_phantom_4": "DJI Phantom 4", "Parrot_disco": "Parrot Disco", "Parrot_mambo_video": "Parrot Mambo",
        "Yuneec_typhoon_h": "Yuneec Typhoon H"}
DD = {"AIR": "DJI Mavic Air", "DIS": "Parrot Disco", "INS": "DJI Inspire 2", "MIN": "DJI Mavic Mini", "MP1": "DJI Mavic Pro", "MP2": "DJI Mavic Pro 2", "PHA": "DJI Phantom 4"}
ORDER = ["DJI Mini 5 Pro", "DJI Mini 4 Pro", "DJI Mini 3 Pro", "DJI Mini 3", "DJI Mavic Mini", "DJI Air 3S", "DJI Air 2S", "DJI Mavic Air 2", "DJI Mavic Air",
         "DJI Mavic 3", "DJI Mavic 3 Classic", "DJI Mavic Pro", "DJI Mavic Pro 2", "DJI Phantom 4", "DJI Inspire 2", "DJI Matrice 100", "DJI Matrice 210",
         "Parrot Disco", "Parrot Mambo", "Yuneec Typhoon H"]
SRC = [("DRFF-R2", "최신 DJI 8기종 실내 녹음(공개, IQ 100 MS/s)"), ("Tampere", "DJI·Yuneec 등 옛 기종 무반사실 녹음(공개)"),
       ("DroneDetect", "DJI 옛 기종 녹음(공개, 간섭 없는 녹음만)"), ("RFUAV", "DJI Mini 4 Pro 녹음(공개)")]


def kind(g):
    s = g.split()[0]
    if s == "DRFF": return DRFF[re.match(r"DRFF (\w+?)_\d+_c\d", g).group(1)], "DRFF-R2"
    if s == "Tampere": return TAMP[re.match(r"Tampere (\w+?)_2G", g).group(1)], "Tampere"
    if s == "DroneDetect": return DD[re.search(r"CLEAN/([A-Z0-9]+)_", g).group(1)], "DroneDetect"
    return ("DJI Mini 3" if "MINI3/" in g else "DJI Mini 4 Pro"), "RFUAV"


def counts():
    m = json.load(open(ROOT / "runs/iq_pieces_v7_meta.json")); by_kind = collections.defaultdict(collections.Counter); by_src = collections.defaultdict(collections.Counter)
    srcs = collections.defaultdict(set)
    for r in m:
        k, s = kind(r["group"]); by_kind[k][r["part"]] += 1; by_src[s][r["part"]] += 1; srcs[k].add(s)
    n2 = collections.Counter(r["part"] for r in json.load(open(ROOT / "runs/neo2_pieces_meta.json")) if r["s0_db"] >= 1.5)
    assert set(by_kind) == set(ORDER)
    return by_kind, by_src, srcs, n2


def section(neo2=True, level=4):
    by_kind, by_src, srcs, n2 = counts(); h = "#" * level
    L = [f"{h} 합성 그림에 넣은 드론 조각의 출처 (조각 수)", "", "| 출처 | 무엇 | 학습용 조각 | 검증용 조각 |", "|---|---|---:|---:|"]
    L += [f"| {s} | {d} | {by_src[s]['train']} | {by_src[s]['val']} |" for s, d in SRC]
    if neo2: L.append(f"| 공원 Neo 2 | 팀원이 HackRF 로 받은 DJI Neo 2 녹음(학습 공원 01·03·09, 검증 공원 10) | {n2['train']:,} | {n2['val']:,} |")
    L += ["", "- 드론 조각은 드론 신호만 있는 녹음에서, 드론 신호의 주파수 범위만 남기고 20 ms 씩 잘라 낸 IQ 다. 합성할 때 조각을 그림 안 아무 주파수에나 옮겨 넣는다.",
          "- 공개 데이터 네 곳의 조각은 모델 1·2·3·1a·2a·3a 가 모두 같다. 공원 Neo 2 조각은 모델 1a·2a·3a 에만 들어 있다(report_261010)." if neo2 else
          "- 이 표의 조각은 모델 1·2·3 의 학습·검증에 쓴 것이다. DJI Neo 2 조각은 들어 있지 않다(Neo 2 조각은 뒤에 만든 모델 1a·2a·3a 에만 넣음, report_261010).",
          "", f"{h} 드론 조각의 기종 (조각 수)", "", "| 기종 | 출처 | 학습 조각 | 검증 조각 | 테스트 조각 |", "|---|---|---:|---:|---:|"]
    T = collections.Counter()
    for k in ORDER:
        L.append(f"| {k} | {', '.join(s for s, _ in SRC if s in srcs[k])} | {by_kind[k]['train']} | {by_kind[k]['val']} | {by_kind[k]['test']} |"); T.update(by_kind[k])
    if neo2: L.append(f"| DJI Neo 2 (모델 1a·2a·3a 만) | 팀원의 공원 HackRF 녹음 | {n2['train']:,} | {n2['val']:,} | 0 |")
    L.append(f"| **합계{' (Neo 2 빼고)' if neo2 else ''}** | | **{T['train']}** | **{T['val']}** | **{T['test']}** |")
    L += ["", "- 테스트 조각은 학습·검증 조각과 다른 녹음(다른 실물 드론이나 다른 비행 방식, 같은 녹음이면 다른 시간 범위)에서 잘라 합성 테스트에만 썼다. DJI Mini 3 와, 영상 폭을 20 MHz 로 설정한 DJI Mini 4 Pro 녹음(테스트 13개)은 학습·검증에 없다.",
          "- DJI Mavic Pro 처럼 조각이 적은 기종은, 신호가 켜졌을 때와 꺼졌을 때의 세기 차이가 충분한 구간만 조각으로 잘랐기 때문이다.",
          "- DRFF-R2 의 \"Air 2\"(파일 이름 mavicAir2)는 DJI 공식 이름으로 Mavic Air 2 다.", ""]
    return "\n".join(L) + "\n"


INTRO_261009 = """### 참고 — 모델 1·2·3 의 학습에 쓴 드론 조각

모델 1·2·3 은 아래 드론 조각을 넣은 합성 그림으로 배웠다(report_261006-08). DJI Neo 2 조각은 없다. 이 보고서에서 모델 2·3 이 Neo 2 를 대부분 놓친 것은 이 때문으로 보인다(3.4절).

"""
PARK_APPENDIX = """## 부록 E. 참고 — 이 녹음으로 만든 드론 조각 (모델 학습용)

이 분석 뒤에, 공원 녹음 01·03·09(학습)와 10(검증)에서 Neo 2 신호가 보이는 20 ms 를 잘라 드론 조각을 만들어 모델 1a·2a·3a 의 합성 학습 그림에 넣었다(report_261010). 나머지 세션(02 는 바탕 잡음, 04–08·11–13 은 테스트)은 조각으로 쓰지 않았다. 아래는 그 조각과 공개 데이터 조각의 출처·기종이다.
"""


def insert_before(doc, anchor, text):
    assert doc.count(anchor) == 1, (anchor, doc.count(anchor)); i = doc.index(anchor); return doc[:i] + text + "\n" + doc[i:]


if __name__ == "__main__":
    print(section(True)); print(section(False))
