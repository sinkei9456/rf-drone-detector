"""논문 본문 그림 (2026-11-17). 그림 안 글자는 한국어(나눔고딕).
그림 안 이름은 원고와 같다: 생성 간섭 모델(2 %·20 %·50 %·80 % 조건), 실제 배경 모델(객차 배경 모델, 수원역 배경 모델),
미학습 계열 평가, Neo 2 합성 평가, 저간섭 실측 평가, 혼잡 장소 실측 평가.
그림 번호는 rev2 본문에 처음 나오는 순서를 따른다.
  concept   : rev2_f1_concept.png   그림 1. 도심 배치 개념과 처리 흐름 (rev1 그림 2와 같은 그림)
  design    : rev2_f2_design.png    그림 2. 실험 설계 (rev1 그림 1의 글자를 새 이름으로)
  picture   : rev2_f3_input.png     그림 3. 입력 이미지 (rev1 그림 3과 같은 그림)
  synthesis : rev2_f4_synthesis.png 그림 4. 합성 이미지 한 장을 만드는 과정 (rev1 그림 4와 같은 그림)
  density   : rev2_f5_density.png   그림 5. 간섭 밀도 네 조건의 예 (칸 제목을 밀도 숫자로)
  그림 6(rev2_f6_realbg_curves.png)은 python paper/src/fig_exp2_curves.py --rev2 가 만든다.
사용: python paper/src/fig_rev2.py [이름 ...]   (이름 없이 실행하면 전부)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
import fig_rev1 as R
from fig_rev1 import card, arrow, BLUE, AMBER, TEAL, PURPLE, INKD, MUTED, LINE, PALE

RENAME = {"rev1_f2_concept.png": "rev2_f1_concept.png", "rev1_f3_input.png": "rev2_f3_input.png",
          "rev1_f4_synthesis.png": "rev2_f4_synthesis.png"}
_save = R.save


def _run_renamed(fn):
    """rev1 의 그림 함수를 그대로 돌리고 파일 이름만 rev2 번호로 바꿔 저장한다."""
    R.save = lambda fig, name, **k: _save(fig, RENAME.get(name, name), **k)
    try: fn()
    finally: R.save = _save


def concept(): _run_renamed(R.concept)
def picture(): _run_renamed(R.picture)
def synthesis(): _run_renamed(R.synthesis)


def design():
    """그림 2. 위 줄 = 생성 간섭 모델(모델 40개), 아래 줄 = 실제 배경 모델(장소마다 모델 2개)."""
    fig, ax = plt.subplots(figsize=(15, 10.4)); ax.set(xlim=(0, 1), ylim=(0, 1.42)); ax.axis("off"); o = .38
    for x, t in [(.02, "01  데이터"), (.355, "02  합성 데이터와 학습"), (.69, "03  평가")]: ax.text(x, 1.418, t, fontsize=13, weight="bold", color=BLUE, va="top")
    ax.text(.02, 1.362, "생성 간섭 모델  |  간섭이 적은 공원 배경에 드론 신호와 생성한 간섭을 합성", fontsize=12, weight="bold", color=INKD, va="top")
    card(ax, .02, .61 + o, .29, .30, "공개 드론 IQ 데이터", "20개 기종  ·  드론 신호 세그먼트 303개\n제품 계열별 5개 fold\n4개 fold 학습  /  1개 fold 평가\n평가 fold 기종은 학습에 쓰지 않음\n평가 fold를 바꿔 5회 반복", BLUE, PALE)
    card(ax, .02, .33 + o, .29, .22, "공원 배경", "합성 이미지의 배경 잡음\n0–1 s 학습  /  1–2 s 검증\n2–3 s 합성 평가", TEAL, PALE)
    card(ax, .02, .05 + o, .29, .22, "DJI Neo 2", "학습·검증에 쓰지 않음\n평가에만 사용", PURPLE, "#f8f6fb")
    card(ax, .355, .61 + o, .29, .30, "합성 데이터", "드론 신호 세그먼트 + 공원 배경\n+ 생성한 와이파이·블루투스 버스트\n학습 6,000장  /  검증 600장\nIQ 파일 단위로 분할", TEAL, PALE)
    card(ax, .355, .33 + o, .29, .22, "학습 조건 4개 (간섭 밀도)", "2 %  /  20 %  /  50 %  /  80 %\n스펙트럼 점유율 중앙값 기준", BLUE)
    card(ax, .355, .05 + o, .29, .22, "YOLO26s 학습", "4개 조건 × 5개 fold × 2개 시드\n= 모델 40개\n판정 임계값: 검증 데이터에서 결정", BLUE, "#f1f6fb")
    card(ax, .69, .74 + o, .29, .17, "미학습 계열 평가 (합성)", "평가 fold의 드론 신호 세그먼트\n+ 생성한 간섭, 간섭 밀도 4단계", AMBER, "#fffaf6", 9.5)
    card(ax, .69, .51 + o, .29, .19, "Neo 2 합성 평가", "Neo 2 드론 신호 세그먼트\n+ 생성한 간섭, 간섭 밀도 4단계", PURPLE, "#faf8fc", 9.5)
    card(ax, .69, .28 + o, .29, .19, "저간섭 실측 평가", "공원·실내 IQ 데이터\nNeo 2 전원 켜짐과 꺼짐", PURPLE, "#faf8fc", 9.5)
    card(ax, .69, .05 + o, .29, .19, "혼잡 장소 실측 평가", "수원역 대합실: 비비행 상태 영상 전송, 비행\n전철 1호선 객차: 비비행 상태 영상 전송\n드론 전원을 끈 IQ 데이터로 FPR", PURPLE, "#faf8fc", 9.5)
    arrow(ax, (.31, .77 + o), (.35, .77 + o))
    ax.plot([.31, .333, .333], [.44 + o, .44 + o, .66 + o], color=MUTED, lw=1.4); arrow(ax, (.333, .66 + o), (.35, .66 + o))
    arrow(ax, (.5, .61 + o), (.5, .555 + o)); arrow(ax, (.5, .33 + o), (.5, .275 + o))
    ax.plot([.645, .665], [.16 + o, .16 + o], color=MUTED, lw=1.2); ax.plot([.665, .665], [.145 + o, .825 + o], color=MUTED, lw=1.2)
    for y in (.825, .605, .375, .145): arrow(ax, (.665, y + o), (.686, y + o))
    ax.plot([.02, .98], [.405, .405], color=LINE, lw=1.2, ls="--")
    ax.text(.02, .385, "실제 배경 모델  |  혼잡한 장소의 실제 배경에 드론 신호를 합성", fontsize=12, weight="bold", color=INKD, va="top")
    card(ax, .02, .02, .29, .30, "혼잡 장소의 드론 전원 꺼짐 IQ 데이터", "같은 날 두 장소에서 취득, 장소별로 사용\n객차: 세션 6개 학습, 2개 검증\n수원역: 영상 전송 전 학습,\n영상 전송 후 검증", TEAL, PALE)
    card(ax, .355, .02, .29, .30, "실제 배경 + 드론 신호", "공개 20개 기종의 드론 신호 세그먼트\n+ 실제 혼잡 배경 (간섭을 더하지 않음)\n드론 한 클래스, 시드 2개, 판정 임계값은\n학습 장소의 검증 데이터에서 결정", BLUE, "#f1f6fb")
    card(ax, .69, .19, .29, .13, "객차 배경 모델", "수원역 데이터로 평가", PURPLE, "#faf8fc", 9.5)
    card(ax, .69, .02, .29, .13, "수원역 배경 모델", "객차 데이터로 평가", PURPLE, "#faf8fc", 9.5)
    arrow(ax, (.31, .17), (.35, .17)); ax.plot([.645, .665], [.17, .17], color=MUTED, lw=1.2); ax.plot([.665, .665], [.085, .255], color=MUTED, lw=1.2)
    for y in (.255, .085): arrow(ax, (.665, y), (.686, y))
    ax.text(.69, -.025, "보라색 = DJI Neo 2 데이터(평가에만 사용)", fontsize=9.5, color=PURPLE)
    _save(fig, "rev2_f2_design.png", dpi=200, bbox_inches="tight", pad_inches=.12)


def density():
    """그림 5. rev1 그림 5와 같은 이미지·같은 버스트, 칸 제목만 밀도 숫자로 부른다."""
    arr, meta = R.pieces(); bg = R.bg_iq(R.PARK, off=7_000_000); z = np.zeros_like(bg); S = R.S
    j = next(k for k, m in enumerate(meta) if m["group"] == "DRFF mavic3_1_c1"); p = dict(meta[j]); p["iq"] = np.array(arr[j])
    xd, bd, _ = S.add_drone(z, p, np.random.default_rng(3), 12.0, fo=0.0)
    W = [R.BG_ORIG(np.random.default_rng([41, i]), 1, 0, base=z)[:2] for i in range(80)]
    B = [R.BG_ORIG(np.random.default_rng([43, i]), 0, 1, base=z)[:2] for i in range(240)]
    W = [w for w in W if w[1]]; B = [b for b in B if b[1]]

    def build(k):
        x = bg + xd + sum((w[0] for w in W[:k]), z) + sum((b[0] for b in B[:3 * k]), z)
        return x.astype(np.complex64), None
    targets = [("2 % 조건", None), ("20 % 조건", 0.20), ("50 % 조건", 0.50), ("80 % 조건", 0.80)]; out = []; k = 1
    for name, tg in targets:
        if tg is None:
            x, _ = build(k); out.append((name, k, R.occupancy(x, bg + xd, [bd]), x)); continue
        while True:
            x, _ = build(k); o = R.occupancy(x, bg + xd, [bd])
            if o >= tg - 0.01 or k >= len(W) or 3 * k >= len(B): break
            k += 1
        out.append((name, k, o, x))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.9))
    for i, (a, (name, kk, o, x)) in enumerate(zip(ax, out)):
        R.show(a, S.to_gray_v7(x), yl=i == 0); R.draw_boxes(a, [bd])
        a.set_title(f"{name}\n와이파이 {kk}개, 블루투스 {3 * kk}개, 스펙트럼 점유율 {100 * o:.1f} %", fontsize=10)
    fig.tight_layout(); _save(fig, "rev2_f5_density.png")
    print([(n, kk, round(100 * o, 1)) for n, kk, o, x in out])


ALL = ["concept", "design", "picture", "synthesis", "density"]
if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL): globals()[k]()
