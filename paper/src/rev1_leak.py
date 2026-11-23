"""생성 간섭 모델의 데이터 누출(data leakage) 점검 (2026-11-12). 합성 데이터를 만든 뒤, 학습 전에 돌린다. 하나라도 실패하면 REV1_LEAK_OK 를 쓰지 않는다.
  1. 평가 fold의 기종과 Neo 2 의 드론 신호 세그먼트가 학습·검증 데이터에 없다.
  2. 한 IQ 파일(group)의 드론 신호 세그먼트가 train 과 validation 양쪽에 들어가지 않는다. 미학습 계열 평가의 IQ 파일도 학습·검증 데이터에 없다.
  3. 공원 배경: train 은 Park02_1(0–1 s), validation 은 Park02_2(1–2 s), 테스트는 Park02_3(2–3 s)만 쓴다.
  4. 미학습 계열 평가·드론 없는 평가 이미지의 난수 씨앗이 학습·검증 데이터와 겹치지 않는다(간섭 신호를 다른 seed 로 생성).
  6. 학습 목록(학습·검증 데이터)에 실측 이미지(data/derived/hackrf)와 평가 이미지가 없고, train 과 validation 목록이 겹치지 않으며, 목록의 파일이 모두 있다.
  (5 threshold·마지막 epoch 는 평가 코드 rev1_eval.py 가 지킨다. 7 은 이 점검 자체.)
출력: paper/runs/rev1/leak_check.md
사용: python paper/src/rev1_leak.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C

meta = C.pieces()[1]; model = [C.model_of(m) for m in meta]; L = ["# 생성 간섭 모델의 데이터 누출(data leakage) 점검", ""]; bad = []


def check(ok, msg):
    L.append(f"- {'통과' if ok else '**실패**'}: {msg}"); (None if ok else bad.append(msg))


def load(name): return json.load(open(C.DATA / name / "meta.json"))


test_seeds = set()
for fk, fname, models in C.FOLDS:
    for cond, _, _ in C.CONDS:
        t = load(C.test_name(fk, cond)); test_seeds |= {tuple(r["seed"]) for r in t}
for cond, _, _ in C.CONDS: test_seeds |= {tuple(r["seed"]) for r in load(C.nodrone_name(cond))}

for fk, fname, models in C.FOLDS:
    L += ["", f"## {fname} fold ({fk})", ""]
    for cond, _, cname in C.CONDS:
        t = load(C.test_name(fk, cond)); tg = {p["group"] for r in t for p in r["pieces"]}; tm = {model[p["piece"]] for r in t for p in r["pieces"]}
        check(tm <= set(models), f"{cname} 미학습 계열 평가: 드론 기종이 모두 평가 fold 기종 ({len(tm)}개 기종, IQ 파일 {len(tg)}개)")
        check({r["bg"] for r in t} == {C.BG["test"]}, f"{cname} 미학습 계열 평가: 배경 잡음은 {C.BG['test']}만")
        for rep in C.REPS:
            name = C.set_name(fk, cond, rep); info = load(name); tr = [r for r in info if r["stem"].startswith("train")]; va = [r for r in info if r["stem"].startswith("val")]
            gt = {p["group"] for r in tr for p in r["pieces"]}; gv = {p["group"] for r in va for p in r["pieces"]}
            mt = {model[p["piece"]] for r in tr + va for p in r["pieces"]}
            check(not (mt & set(models)), f"{name}: 학습·검증 데이터에 평가 fold 기종 없음")
            check(not any("neo" in g.lower() for g in gt | gv), f"{name}: Neo 2 드론 신호 세그먼트 없음")
            check(not (gt & gv), f"{name}: 학습·검증 데이터에 함께 들어간 IQ 파일 {len(gt & gv)}개")
            check(not ((gt | gv) & tg), f"{name}: 미학습 계열 평가의 IQ 파일이 학습·검증 데이터에 {len((gt | gv) & tg)}개")
            check({r["bg"] for r in tr} == {C.BG["train"]} and {r["bg"] for r in va} == {C.BG["val"]}, f"{name}: 배경 잡음 train {C.BG['train']}, validation {C.BG['val']}")
            sd = {tuple(r["seed"]) for r in tr + va}
            check(not (sd & test_seeds), f"{name}: 평가 이미지와 같은 난수 씨앗 {len(sd & test_seeds)}개")
            lt = [l.strip() for l in open(C.RUNS / "lists" / f"{name}_train.txt") if l.strip()]
            lv = [l.strip() for l in open(C.RUNS / "lists" / f"{name}_val.txt") if l.strip()]
            check(len(lt) == C.N_TRAIN and len(lv) == C.N_VAL and not (set(lt) & set(lv)), f"{name}: 목록 train {len(lt)}장, validation {len(lv)}장, 겹침 {len(set(lt) & set(lv))}장")
            check(all(f"/rev1/{name}/images/" in p for p in lt + lv), f"{name}: 목록에 이 데이터셋의 합성 이미지만 있음(실측·평가 이미지 없음)")
            miss = [p for p in lt + lv if not Path(p).exists() or not Path(p.replace("/images/", "/labels/")).with_suffix(".txt").exists()]
            check(not miss, f"{name}: 목록의 이미지·라벨 파일 모두 있음 (없는 것 {len(miss)}개)")
for cond, _, cname in C.CONDS:
    t = load(C.nodrone_name(cond))
    check(all(r["n_drone"] == 0 for r in t) and {r["bg"] for r in t} == {C.BG["test"]}, f"{cname} 드론 없는 평가 이미지: 드론 0개, 배경 잡음 {C.BG['test']}만")
L.insert(2, f"**결과: {'모두 통과' if not bad else f'실패 {len(bad)}개'}** (검사 {sum(l.startswith('- ') for l in L)}개)")
(C.RUNS / "leak_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L[:3])); [print("실패:", b) for b in bad]
if not bad: print("REV1_LEAK_OK")
