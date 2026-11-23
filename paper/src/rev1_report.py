"""생성 간섭 모델 40개의 결과 모으기 (2026-11-13). 원고 6장 표 5–10 에 들어갈 값.
입력: paper/runs/rev1/eval/<이름>/result.json(판정 임계값, 미학습 계열 평가, 저간섭 실측 평가; paper/src/rev1_eval.py), test1_2.json(Neo 2 합성 평가; paper/src/rev1_test1_2.py)
  이름 = <fold>_<학습 조건>_s<시드>. 학습 조건 = c02(2 % 조건), c20(20 % 조건), c50(50 % 조건), c80(80 % 조건).
집계: 표의 값은 fold 마다 그 fold 모델(seed)의 평균을 낸 뒤 fold 들을 같은 무게로 평균한 값(학습이 덜 끝나 seed 수가 fold 마다 달라도 한 fold 가 더 무겁지 않게).
  칸마다 쓴 모델 수를 함께 적고, 범위(모델 중 가장 낮은 값–가장 높은 값)는 따로 표로 둔다. --seed 1 처럼 주면 그 seed 의 모델만 쓴다(학습 중간 보고용).
  저간섭 실측 평가의 세션별 재현율·FPR 은 모델마다 그 세션 이미지 전부로 계산한 값의 평균이다.
출력: paper/runs/rev1/RESULTS.md(사람이 읽는 표), paper/runs/rev1/results.json(원고에 옮길 값)
사용: python paper/src/rev1_report.py [--seed 1]   (--seed 를 주면 RESULTS_seed1.md, results_seed1.json)"""
import json, re, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C

TRAIN = [("c02", "2 %"), ("c20", "20 %"), ("c50", "50 %"), ("c80", "80 %")]                     # 원고 표 5–10 의 행 이름
TEST = [(c, f"{t * 100:.0f} %") for c, t, _ in C.CONDS]
FOLDS = [(k, n) for k, n, _ in C.FOLDS]
REC = [("01", "공원, 비비행, 조종기와 1 m (01)"), ("03", "공원, 비행, 조종기와 10 m (03)"), ("09", "공원, 고도 10 m (09)"), ("10", "공원, 고도 10 m, 수신기가 조종기에서 8 m (10)"),
       ("13", "공원, 근처 자유 비행 (13)"), ("07", "공원, 고도 30 m (07)"), ("indoor_ON", "실내, 비비행, 2–3 m")]          # 원고 표 9 의 열 이름
REF = [("04", "공원 세션 04"), ("05", "공원 세션 05"), ("06", "공원 세션 06"), ("08", "공원 세션 08"), ("11", "공원 세션 11"), ("12", "공원 세션 12")]
FPR = [("02", "공원, 드론·조종기 전원 꺼짐 (세션 02의 2–3 s)"), ("indoor_OFF", "실내, Neo 2 전원 꺼짐")]                    # 원고 표 10 의 열 이름


def load(seed=None):
    out = []
    for f in sorted((C.RUNS / "eval").glob("*/result.json")):
        m = re.match(r"^(f\d_[a-z]+)_(c02|c20|c50|c80)_s(\d)$", f.parent.name)
        if not m: continue
        if seed and int(m[3]) != seed: continue
        r = json.load(open(f)); t2 = f.parent / "test1_2.json"
        out.append(dict(name=f.parent.name, fold=m[1], train=m[2], seed=int(m[3]), res=r, t2=json.load(open(t2)) if t2.exists() else None))
    return out


def pct(v): return "" if v is None else f"{v * 100:.1f}"


def agg(R, train, get):
    """반환 (fold 균형 평균, 가장 낮은 값, 가장 높은 값, 모델 수) 또는 None."""
    v = [(r["fold"], x) for r in R if r["train"] == train for x in [get(r)] if x is not None]
    if not v: return None
    fm = [np.mean([x for f, x in v if f == fk]) for fk in sorted({f for f, _ in v})]
    return (float(np.mean(fm)), float(min(x for _, x in v)), float(max(x for _, x in v)), len(v))


def table(title, R, get, cols, rows=TRAIN, rng=False):
    L = [title, "", "| 학습 조건 | " + " | ".join(n for _, n in cols) + " | 모델 수 |", "|---|" + "---:|" * (len(cols) + 1)]
    for tc, tn in rows:
        a = [agg(R, tc, lambda r, c=c: get(r, c)) for c, _ in cols]
        if all(x is None for x in a): L.append(f"| {tn} | " + " | ".join("평가 전" for _ in cols) + " | 0 |"); continue
        cell = (lambda x: "" if x is None else f"{x[1] * 100:.1f}–{x[2] * 100:.1f}") if rng else (lambda x: "" if x is None else pct(x[0]))
        L.append(f"| {tn} | " + " | ".join(cell(x) for x in a) + f" | {max(x[3] for x in a if x)} |")
    return L + [""]


def main(seed=None):
    R = load(seed); J = dict(made=time.strftime("%Y-%m-%d %H:%M"), seed=seed, models=[r["name"] for r in R]); sfx = f"_seed{seed}" if seed else ""
    n_by = {tc: sum(r["train"] == tc for r in R) for tc, _ in TRAIN}
    L = ["# 생성 간섭 모델 결과", "", f"paper/src/rev1_report.py 가 {J['made']} 에 만든 표{f' (시드 {seed} 모델만)' if seed else ''}. 평가가 끝난 모델 {len(R)}개(학습 조건별: " + ", ".join(f"{tn} 조건 {n_by[tc]}개" for tc, tn in TRAIN) + ").",
         "값은 fold 마다 시드 평균을 낸 뒤 fold 를 같은 무게로 평균한 값, 단위는 모두 %. 모델 하나의 판정은 자기 검증 데이터에서 정한 판정 임계값을 쓴다.", "",
         "## 1. 미학습 계열 평가 (합성)", "",
         "평가 fold 기종의 드론 신호 세그먼트를 공원 배경 2–3 s 와 간섭 밀도 4단계(2·20·50·80 %)로 합성한 이미지. 드론 존재 이미지 fold 마다 500장, 드론 부재 이미지 500장(공통).", ""]
    g1 = lambda k: (lambda r, c: r["res"]["test1"][c][k])
    L += table("**표 1-1. 미학습 계열 평가의 재현율 (단위: %, 행 = 학습 조건, 열 = 평가 이미지의 간섭 밀도)**", R, g1("recall"), TEST)
    L += table("**표 1-2. 미학습 계열 평가의 FPR (단위: %)**", R, g1("fpr"), TEST)
    L += table("**표 1-3. 미학습 계열 평가 재현율의 범위 (단위: %, fold × 시드 중 가장 낮은 값–가장 높은 값)**", R, g1("recall"), TEST, rng=True)
    L += table("**표 1-4. 미학습 계열 평가 FPR의 범위 (단위: %)**", R, g1("fpr"), TEST, rng=True)
    J["test1"] = {tc: {c: dict(recall=agg(R, tc, lambda r, c=c: r["res"]["test1"][c]["recall"]), fpr=agg(R, tc, lambda r, c=c: r["res"]["test1"][c]["fpr"])) for c, _ in TEST} for tc, _ in TRAIN}
    L += ["### fold 별 미학습 계열 평가", ""]
    for i, (fk, fn) in enumerate(FOLDS):
        Rf = [r for r in R if r["fold"] == fk]
        if not Rf: continue
        L += table(f"**표 1-{5 + 2 * i}. {fn} fold 미학습 계열 평가의 재현율 (단위: %)**", Rf, g1("recall"), TEST)
        L += table(f"**표 1-{6 + 2 * i}. {fn} fold 미학습 계열 평가의 FPR (단위: %)**", Rf, g1("fpr"), TEST)
    J["test1_fold"] = {fk: {tc: {c: dict(recall=agg([r for r in R if r["fold"] == fk], tc, lambda r, c=c: r["res"]["test1"][c]["recall"]),
                                          fpr=agg([r for r in R if r["fold"] == fk], tc, lambda r, c=c: r["res"]["test1"][c]["fpr"])) for c, _ in TEST} for tc, _ in TRAIN} for fk, _ in FOLDS}

    L += ["## 2. Neo 2 합성 평가", "", "Neo 2 드론 신호 세그먼트(짧은 버스트 = 공원 01, 긴 버스트 = 공원 03·09·10)를 미학습 계열 평가와 같은 규칙으로 합성한 이미지. 간섭 밀도마다 짧은 버스트 250장, 긴 버스트 250장. FPR 은 미학습 계열 평가와 같다(드론 부재 이미지 공통).", ""]
    g2 = lambda s: (lambda r, c: r["t2"]["recall"][c][s] if r["t2"] else None)
    L += table("**표 2-1. Neo 2 합성 평가의 재현율, 짧은 버스트 (단위: %)**", R, g2("short"), TEST)
    L += table("**표 2-2. Neo 2 합성 평가의 재현율, 긴 버스트 (단위: %)**", R, g2("long"), TEST)
    J["test2"] = {tc: {c: {s: agg(R, tc, lambda r, c=c, s=s: r["t2"]["recall"][c][s] if r["t2"] else None) for s in ("short", "long", "all")} for c, _ in TEST} for tc, _ in TRAIN}
    L += ["### fold 별 Neo 2 합성 평가", "", "fold 마다 학습에서 빠진 계열이 다르다. Neo 2 와 신호 모양이 비슷한 기종(보충 자료 표 S8): 짧은 버스트 = Mini 4 Pro(Mini fold), 긴 버스트 = Mavic 3(Mavic fold)·Phantom 4(전문가용 fold).", ""]
    for i, (fk, fn) in enumerate(FOLDS):
        Rf = [r for r in R if r["fold"] == fk]
        if not Rf: continue
        L += table(f"**표 2-{3 + 2 * i}. {fn} fold 를 뺀 모델의 Neo 2 합성 평가 재현율, 짧은 버스트 (단위: %)**", Rf, g2("short"), TEST)
        L += table(f"**표 2-{4 + 2 * i}. {fn} fold 를 뺀 모델의 Neo 2 합성 평가 재현율, 긴 버스트 (단위: %)**", Rf, g2("long"), TEST)
    J["test2_fold"] = {fk: {tc: {c: {s: agg([r for r in R if r["fold"] == fk], tc, lambda r, c=c, s=s: r["t2"]["recall"][c][s] if r["t2"] else None) for s in ("short", "long")} for c, _ in TEST} for tc, _ in TRAIN} for fk, _ in FOLDS}

    L += ["## 3. 저간섭 실측 평가", "", "2026-10-08 공원과 실내에서 HackRF One(수신 필터 20 MHz)으로 취득한 Neo 2 실측 IQ 데이터. 드론 채널 밖 스펙트럼 점유율 0.1–0.9 %. 세션 목록은 보충 자료 표 S12.", ""]
    gb = lambda r, s: r["res"]["baseline_recall"][s]["all"] if s in r["res"]["baseline_recall"] else None
    L += table("**표 3-1. 저간섭 실측 평가의 재현율, 세션별 (단위: %, 열 이름의 괄호는 공원 세션 번호)**", R, gb, REC)
    L += table("**표 3-2. 저간섭 실측 평가의 재현율, 고도 약 50 m 세션 (참고, 신호가 이미지에서 거의 안 보임, 단위: %)**", R, gb, REF)
    gf = lambda r, s: r["res"]["baseline_fpr"][s]["fpr"] if s in r["res"]["baseline_fpr"] else None
    L += table("**표 3-3. 저간섭 실측 평가의 FPR, 세션별 (단위: %)**", R, gf, FPR)
    J["baseline_recall"] = {tc: {s: agg(R, tc, lambda r, s=s: gb(r, s)) for s, _ in REC + REF} for tc, _ in TRAIN}
    J["baseline_fpr"] = {tc: {s: agg(R, tc, lambda r, s=s: gf(r, s)) for s, _ in FPR} for tc, _ in TRAIN}

    L += ["## 4. 판정 임계값과 검증 데이터", "", "**표 4-1. 검증 데이터에서 정한 판정 임계값과 그때의 이미지 단위 F1 점수 (판정 임계값은 신뢰도 값, F1 점수는 0–1)**", "",
          "| 학습 조건 | 판정 임계값 중앙값 | 판정 임계값 범위 | 검증 데이터 F1 점수 평균 | 모델 수 |", "|---|---:|---|---:|---:|"]
    for tc, tn in TRAIN:
        q = [r for r in R if r["train"] == tc]
        if not q: L.append(f"| {tn} | 평가 전 | | | 0 |"); continue
        th = [r["res"]["val"]["threshold"] for r in q]; f1 = [r["res"]["val"]["f1"] for r in q]
        L.append(f"| {tn} | {np.median(th):.2f} | {min(th):.2f}–{max(th):.2f} | {np.mean(f1):.3f} | {len(q)} |")
    L += ["", "## 5. 시드 1과 시드 2의 차이", "", "**표 5-1. 같은 fold·학습 조건에서 시드 1과 시드 2의 미학습 계열 평가 값 차이 (절댓값 평균, 단위: %p)**", "",
          "| 학습 조건 | 비교한 쌍 수 | 재현율 차이 | FPR 차이 |", "|---|---:|---:|---:|"]
    for tc, tn in TRAIN:
        pairs = [(a, b) for a in R for b in R if a["train"] == b["train"] == tc and a["fold"] == b["fold"] and a["seed"] == 1 and b["seed"] == 2]
        if not pairs: L.append(f"| {tn} | 0 | | |"); continue
        dr = [abs(a["res"]["test1"][c]["recall"] - b["res"]["test1"][c]["recall"]) for a, b in pairs for c, _ in TEST]
        df = [abs(a["res"]["test1"][c]["fpr"] - b["res"]["test1"][c]["fpr"]) for a, b in pairs for c, _ in TEST]
        L.append(f"| {tn} | {len(pairs)} | {np.mean(dr) * 100:.1f} | {np.mean(df) * 100:.1f} |")
    fo, fj = C.RUNS / f"RESULTS{sfx}.md", C.RUNS / f"results{sfx}.json"
    fo.write_text("\n".join(L) + "\n", encoding="utf-8"); json.dump(J, open(fj, "w"), ensure_ascii=False, indent=1)
    print(f"모델 {len(R)}개 → {fo.relative_to(C.ROOT)}, {fj.relative_to(C.ROOT)}", flush=True)


if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); main(ap.parse_args().seed)
