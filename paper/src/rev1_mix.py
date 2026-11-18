"""rev1 혼합 조건: 간섭 밀도 2·20·50·80 % 를 1:1:1:1 로 섞은 train·validation 만들기 (2026-11-13).
원고 3.3절(간섭 밀도 조건), 3.5절(학습).
방법: 새로 합성하지 않는다. 네 조건의 같은 번호 이미지는 드론 신호 세그먼트·위치·SNR·배경 노이즈가 같고 간섭 신호의 양만 다르다(paper/src/rev1_synth.py).
  그래서 번호마다 밀도 하나를 골라 그 조건의 이미지를 쓴다. train 6,000장 = 밀도마다 1,500장, validation 600장 = 밀도마다 150장.
  번호와 밀도의 짝은 seed 마다 정해진 난수로 섞는다(train 난수 [41, seed], validation 난수 [42, seed]). fold 다섯 개에 같은 짝을 쓴다.
  → data/derived/rev1/<fold>_cmix_s<seed>/{images,labels} 는 원래 이미지·라벨을 가리키는 상대 링크, meta.json 은 원래 정보 + density(고른 밀도)
  → 학습 목록·yaml(paper/runs/rev1/{lists,yaml}), 점검 결과 paper/runs/rev1/mix_check.md(data leakage 점검은 paper/src/rev1_leak.py 와 같은 항목)
사용: python paper/src/rev1_mix.py   (점검을 모두 통과하면 REV1_MIX_OK)"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C

MIX = "cmix"


def assign(rep):
    """반환 (train 밀도 번호 6,000개, validation 밀도 번호 600개). 밀도 번호 0–3 = C.CONDS 순서."""
    tr = np.random.default_rng([41, rep]).permutation(np.repeat(np.arange(4), C.N_TRAIN // 4))
    va = np.random.default_rng([42, rep]).permutation(np.repeat(np.arange(4), C.N_VAL // 4))
    return tr, va


def link(src, dst):
    if dst.is_symlink() or dst.exists(): dst.unlink()
    os.symlink(os.path.relpath(src, dst.parent), dst)


def build():
    for fk, _, _ in C.FOLDS:
        for rep in C.REPS:
            name = C.set_name(fk, MIX, rep); d = C.DATA / name
            if (d / "meta.json").exists(): print("있음, 건너뜀:", name, flush=True); continue
            (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True)
            src_meta = {cond: {r["stem"]: r for r in json.load(open(C.DATA / C.set_name(fk, cond, rep) / "meta.json"))} for cond, _, _ in C.CONDS}
            tr, va = assign(rep); info = []
            for part, ch in (("train", tr), ("val", va)):
                for i, c in enumerate(ch):
                    cond = C.CONDS[int(c)][0]; stem = f"{part}_{i:05d}"; s = C.DATA / C.set_name(fk, cond, rep)
                    link(s / "images" / f"{stem}.png", d / "images" / f"{stem}.png"); link(s / "labels" / f"{stem}.txt", d / "labels" / f"{stem}.txt")
                    info.append(dict(src_meta[cond][stem], density=cond))
            json.dump(info, open(d / "meta.json.tmp", "w")); (d / "meta.json.tmp").rename(d / "meta.json")
            for part in ("train", "val"):
                (C.RUNS / "lists" / f"{name}_{part}.txt").write_text("".join(f"{d / 'images' / (r['stem'] + '.png')}\n" for r in info if r["stem"].startswith(part)))
            (C.RUNS / "yaml" / f"{name}.yaml").write_text(f"train: ../lists/{name}_train.txt\nval: ../lists/{name}_val.txt\nnames:\n  0: drone\n  1: wifi\n  2: bt\n")
            print(f"{name}: train {sum(r['stem'].startswith('train') for r in info)}장, validation {sum(r['stem'].startswith('val') for r in info)}장", flush=True)


def check():
    meta = C.pieces()[1]; model = [C.model_of(m) for m in meta]; L = ["# rev1 혼합 조건 점검", ""]; bad = []

    def ok(c, msg): L.append(f"- {'통과' if c else '**실패**'}: {msg}"); (None if c else bad.append(msg))
    test_seeds = {tuple(r["seed"]) for fk, _, _ in C.FOLDS for cond, _, _ in C.CONDS for r in json.load(open(C.DATA / C.test_name(fk, cond) / "meta.json"))}
    test_seeds |= {tuple(r["seed"]) for cond, _, _ in C.CONDS for r in json.load(open(C.DATA / C.nodrone_name(cond) / "meta.json"))}
    L += ["| 데이터셋 | train 장수 | validation 장수 | train 밀도별 장수 (2·20·50·80 %) | validation 밀도별 장수 | train 점유율 중앙값 (%) |", "|---|---:|---:|---|---|---:|"]
    rows = []
    for fk, _, models in C.FOLDS:
        tg = {p["group"] for cond, _, _ in C.CONDS for r in json.load(open(C.DATA / C.test_name(fk, cond) / "meta.json")) for p in r["pieces"]}
        for rep in C.REPS:
            name = C.set_name(fk, MIX, rep); info = json.load(open(C.DATA / name / "meta.json"))
            tr = [r for r in info if r["stem"].startswith("train")]; va = [r for r in info if r["stem"].startswith("val")]
            cnt = lambda q: "·".join(str(sum(r["density"] == c for r in q)) for c, _, _ in C.CONDS)
            occ = [r["occ"] for r in tr if r["occ"] is not None]
            L.append(f"| {name} | {len(tr)} | {len(va)} | {cnt(tr)} | {cnt(va)} | {np.median(occ) * 100:.1f} |")
            gt = {p["group"] for r in tr for p in r["pieces"]}; gv = {p["group"] for r in va for p in r["pieces"]}; mt = {model[p["piece"]] for r in tr + va for p in r["pieces"]}
            lt = [l.strip() for l in open(C.RUNS / "lists" / f"{name}_train.txt") if l.strip()]; lv = [l.strip() for l in open(C.RUNS / "lists" / f"{name}_val.txt") if l.strip()]
            rows.append((name, [
                (len(tr) == C.N_TRAIN and len(va) == C.N_VAL and cnt(tr) == "·".join([str(C.N_TRAIN // 4)] * 4) and cnt(va) == "·".join([str(C.N_VAL // 4)] * 4), "장수와 밀도별 장수(1:1:1:1)"),
                (not (mt & set(models)), "train·validation 에 테스트 fold 기종 없음"),
                (not any("neo" in g.lower() for g in gt | gv), "Neo 2 드론 신호 세그먼트 없음"),
                (not (gt & gv), f"train·validation 에 함께 들어간 IQ 파일 {len(gt & gv)}개"),
                (not ((gt | gv) & tg), f"테스트 1 의 IQ 파일이 train·validation 에 {len((gt | gv) & tg)}개"),
                ({r["bg"] for r in tr} == {C.BG["train"]} and {r["bg"] for r in va} == {C.BG["val"]}, "배경 노이즈 train Park02_1, validation Park02_2"),
                (not ({tuple(r["seed"]) for r in tr + va} & test_seeds), "테스트 이미지와 같은 난수 씨앗 없음"),
                (len(lt) == C.N_TRAIN and len(lv) == C.N_VAL and not (set(lt) & set(lv)), f"목록 train {len(lt)}장, validation {len(lv)}장, 겹침 {len(set(lt) & set(lv))}장"),
                (all(Path(p).resolve().exists() and Path(p.replace("/images/", "/labels/")).with_suffix(".txt").resolve().exists() for p in lt + lv), "목록의 이미지·라벨 링크가 모두 실제 파일을 가리킴"),
                (all(any(f"/rev1/{C.set_name(fk, c, rep)}/images/" in str(Path(p).resolve()) for c, _, _ in C.CONDS) for p in lt + lv), "링크가 같은 fold·seed 의 네 조건 이미지만 가리킴(실측·테스트 이미지 없음)")]))
    for name, cs in rows:
        L += ["", f"## {name}", ""]
        for c, msg in cs: ok(c, msg)
    L.insert(2, f"**결과: {'모두 통과' if not bad else f'실패 {len(bad)}개'}**")
    (C.RUNS / "mix_check.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L[:16])); [print("실패:", b) for b in bad]
    if not bad: print("REV1_MIX_OK")


if __name__ == "__main__":
    build(); check()
