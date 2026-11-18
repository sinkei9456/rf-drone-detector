"""그림 6: 2번 묶음(혼잡한 실제 배경으로 학습)과 비교 모델의 threshold별 recall·FPR (2026-11-16). 원고 5.4절.
(a) 수원역으로 평가: 2-1(객차 배경으로 학습, seed 2개), 대조(공원 배경 + 합성 간섭, seed 2개), 기존 혼합 조건(1번 묶음, 모델 10개).
(b) 객차로 평가: 2-2(수원역 배경으로 학습), 대조, 기존 혼합 조건.
입력: paper/runs/rev1/exp2/eval/<이름>/{test1_4.csv, result.json}, paper/runs/rev1/eval/<이름>/{test1_4.csv, result.json}
계산: recall(t) = 그 장소의 드론 있는 이미지 중 max_hit ≥ t, FPR(t) = 드론을 끈 이미지 중 max_conf ≥ t, t = 0.01–0.95.
  2번 묶음과 대조는 seed 2개 평균, 기존 혼합은 fold마다 seed 평균 뒤 fold 5개 평균. 큰 표시 = 모델마다 자기 validation threshold 로 판정한 값의 같은 평균.
  테스트 데이터로 threshold 를 고르지 않는다(곡선은 보고용).
출력: paper/fig/rev1_f6_exp2_curves.png (rev1 이름), --rev2 를 주면 paper/fig/rev2_f6_realbg_curves.png (rev2 이름: 객차 배경 모델, 수원역 배경 모델,
  대조 모델, 생성 간섭 모델의 혼합 조건. 2026-11-17)
사용: python paper/src/fig_exp2_curves.py [--rev2]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C, rev1_test1_4 as T
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.font_manager as fm
from fig_rev1 import nanum
plt.rcParams.update({"font.family": nanum(), "axes.unicode_minus": False, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False})
TS = np.round(np.arange(0.01, 0.951, 0.01), 2); X = C.RUNS / "exp2" / "eval"


def curve(d, v):
    D = pd.read_csv(d / "test1_4.csv"); t = json.load(open(d / "result.json"))["val"]["threshold"]
    p = D[D.group.isin([g for g, _, dr in T.GROUPS if dr and g[0] == v])].max_hit.to_numpy()
    n = D[D.group.isin([g for g, _, dr in T.GROUPS if not dr and g[0] == v])].max_conf.to_numpy()
    return np.array([[(n >= x).mean() for x in TS], [(p >= x).mean() for x in TS]]), np.array([(n >= t).mean(), (p >= t).mean()])


def mean_of(dirs, v, folds=None):
    cs = [curve(d, v) for d in dirs]
    if folds is None: return np.mean([c for c, _ in cs], 0), np.mean([m for _, m in cs], 0), len(cs)
    fk = sorted(set(folds)); pick = lambda k: [i for i, f in enumerate(folds) if f == k]
    return (np.mean([np.mean([cs[i][0] for i in pick(k)], 0) for k in fk], 0), np.mean([np.mean([cs[i][1] for i in pick(k)], 0) for k in fk], 0), len(cs))


def main(rev2=False):
    old = sorted(p.parent for p in (C.RUNS / "eval").glob("*_cmix_s*/test1_4.csv")); folds = [d.name.rsplit("_cmix", 1)[0] for d in old]
    panels = [("S", "(a) 수원역 대합실로 평가", "test2_1", "객차 배경 모델: 객차의 실제 배경으로 학습" if rev2 else "2-1: 객차의 실제 배경으로 학습"),
              ("T", "(b) 전철 1호선 객차로 평가", "test2_2", "수원역 배경 모델: 수원역의 실제 배경으로 학습" if rev2 else "2-2: 수원역의 실제 배경으로 학습")]
    lab_c = "대조 모델: 공원 배경 + 생성한 간섭" if rev2 else "대조: 공원 배경 + 합성 간섭"
    lab_m = "생성 간섭 모델, 혼합 조건" if rev2 else "기존 혼합 조건 (1번 묶음)"
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    for a, (v, title, k, lab) in zip(ax, panels):
        a.plot([0, 100], [0, 100], color="#a8a7a0", lw=1, ls="--", zorder=1)
        rows = [(mean_of([X / f"{k}_s{r}" for r in (1, 2)], v), lab, "#0d366b", "D"),
                (mean_of([X / f"test2_c_s{r}" for r in (1, 2)], v), lab_c, "#86b6ef", "s"),
                (mean_of(old, v, folds), lab_m, "#eb6834", "P")]
        for (c, m, n), name, col, mk in rows:
            a.plot(c[0] * 100, c[1] * 100, color=col, lw=2.2, zorder=2, label=f"{name} (모델 {n}개)")
            a.plot(m[0] * 100, m[1] * 100, marker=mk, ms=11, color=col, mec="white", mew=1.5, ls="none", zorder=3)
        a.set_xlim(-2, 102); a.set_ylim(-2, 102); a.set_xticks(range(0, 101, 20)); a.set_yticks(range(0, 101, 20)); a.grid(color="#e4e3df", lw=0.6)
        a.set_title(title, fontsize=11.5, loc="left"); a.set_xlabel("FPR (%, 드론을 끈 이미지 중 양성 비율)")
        a.legend(loc="lower right", fontsize=9, frameon=False); a.text(98, 30, "점선: 구분하지 못할 때", ha="right", fontsize=8.5, color="#52514e")
    ax[0].set_ylabel("recall (%, 드론 있는 이미지 중 정탐 비율)")
    fig.tight_layout(); out = C.ROOT / ("paper/fig/rev2_f6_realbg_curves.png" if rev2 else "paper/fig/rev1_f6_exp2_curves.png"); fig.savefig(out, dpi=110); print(out.relative_to(C.ROOT))


if __name__ == "__main__":
    main(rev2="--rev2" in sys.argv[1:])
