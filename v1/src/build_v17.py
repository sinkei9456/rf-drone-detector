"""모델 1a·2a·3a 두 번째 학습(10-09 밤 사용자 승인: 약하게 만든 조각 절반을 1–2 dB, 비행 조각 절반의 켜져 있는 시간을 1.5–3.5 ms 로 줄임 — doc/report_261010.md) 학습 목록 만들기와 겹침 확인.
src/build_v16.py 를 경로만 바꿔 복사한 것.
그림: 합성 v10(data/derived/yolo_syn_iq_v10, src/synth_iq_build.py --ver v10) 학습 6,000 / 검증 600.
  모델 1a = 합성 v10 (drone·wifi·bt 박스)                         학습 6,000 / 검증 600
  모델 2a = 합성 v10 (drone·wifi·bt) + "드론 아님" 실제 그림          (드론 아님은 모델 2·3 과 같은 그림: split_v7 의 yolo_neg_v14)
  모델 3a = 합성 v10 의 drone 박스만(yolo_syn_iq_v10_drone, 그림은 바로가기) + "드론 아님" 실제 그림
확인(하나라도 실패하면 BUILD_V17_DONE 을 쓰지 않음): 학습·검증·테스트 그림 겹침 없음, Neo 2 조각이 학습 그림은 공원 01·03·09, 검증 그림은 공원 10 에서만,
  바탕이 학습 그림은 안테나 뺀 01·02·공원 02 앞 2초, 검증 그림은 안테나 뺀 03·공원 02 셋째 1초에서만, 드론 아님 그림은 박스 없음.
출력: data/derived/split_v10/{m1a,m2a,m3a}_{train,val}.txt, data/yaml/v17_{1a,2a,3a}.yaml, runs/review/build_v17.txt"""
import os, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/derived"
SYN = D / "yolo_syn_iq_v10"; SD = D / "yolo_syn_iq_v10_drone"; SP = D / "split_v10"
for d in (SD / "images", SD / "labels", SP): d.mkdir(parents=True, exist_ok=True)
rd = lambda p: [l.strip() for l in open(p) if l.strip()]
lab = lambda p: Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
L = []; ok = True


def check(c, msg):
    global ok
    ok &= bool(c); L.append(("통과 " if c else "실패 ") + msg)


syn = {sp: rd(SYN / f"{sp}.txt") for sp in ("train", "val")}
sd = {}; nb = [0, 0]
for sp, ps in syn.items():
    sd[sp] = []
    for p in ps:
        p = Path(p); li = SD / "images" / p.name
        if not li.exists(): os.symlink(p, li)
        lines = [l for l in open(SYN / "labels" / f"{p.stem}.txt") if l.strip()]; keep = [l for l in lines if l.split()[0] == "0"]
        nb[0] += len(keep); nb[1] += len(lines) - len(keep); (SD / "labels" / f"{p.stem}.txt").write_text("".join(keep)); sd[sp].append(str(li))
L.append(f"합성 v10: 학습 {len(syn['train'])} / 검증 {len(syn['val'])}장. drone 박스만 남긴 세트: drone {nb[0]}개 남김, wifi·bt {nb[1]}개 지움")
neg = {sp: [p for p in rd(D / f"split_v7/a_{sp}.txt") if "/yolo_neg_v14/" in p] for sp in ("train", "val")}
L.append(f"드론 아님 실제 그림(모델 2·3 과 같음): 학습 {len(neg['train'])} / 검증 {len(neg['val'])}장")
out = {"m1a_train": syn["train"], "m1a_val": syn["val"], "m2a_train": syn["train"] + neg["train"], "m2a_val": syn["val"] + neg["val"],
       "m3a_train": sd["train"] + neg["train"], "m3a_val": sd["val"] + neg["val"]}
for k, v in out.items(): (SP / f"{k}.txt").write_text("\n".join(v) + "\n"); L.append(f"{k}: {len(v)}장")
# 확인
meta = json.load(open(SYN / "meta.json")); ms = {r["stem"]: r for r in meta}
for sp, sess in (("train", {"01", "03", "09"}), ("val", {"10"})):
    used = {u["session"] for r in meta if r["split"] == sp for u in r["pieces"] if "neo2" in u}
    check(used <= sess, f"{sp} 그림의 Neo 2 조각 세션 {sorted(used)} ⊂ {sorted(sess)}")
    n2 = sum(1 for r in meta if r["split"] == sp and r.get("n_neo2", 0) > 0); L.append(f"  {sp}: Neo 2 조각이 든 그림 {n2}장 / 드론 든 그림 {sum(1 for r in meta if r['split'] == sp and r['n_drone'] > 0)}장")
    bg = {(r["hackrf"]["cond"], r["hackrf"]["file"]) for r in meta if r["split"] == sp}
    allowed = {("off", "01"), ("off", "02"), ("park", "1"), ("park", "2")} if sp == "train" else {("off", "03"), ("park", "3")}
    check(bg <= allowed, f"{sp} 그림의 바탕 {sorted(bg)} ⊂ {sorted(allowed)}")
    cut = [u for r in meta if r["split"] == sp for u in r["pieces"] if "neo2" in u and u.get("on_ms") is not None]
    L.append(f"  {sp}: 켜져 있는 시간을 줄인 Neo 2 조각 {len(cut)}개(세션 {sorted({u['session'] for u in cut})}), 1–1.5 dB 조각 {sum(1 for r in meta if r['split'] == sp for u in r['pieces'] if 'neo2' in u and u['band_db'] <= 1.5)}개")
    check(all(not u["session"].startswith("01") for u in cut), f"{sp}: 1 m(01) 조각은 자르지 않음")
    old = {u["part"] for r in meta if r["split"] == sp for u in r["pieces"] if "piece" in u}
    check(old <= {sp}, f"{sp} 그림의 기존 드론 조각은 {sp}용만 ({sorted(old)})")
test = set(str(Path(p).resolve()) for f in ("split_v6/syn8_test.txt", "split_v6/drff8_all.txt") for p in rd(D / f))
for m in ("m1a", "m2a", "m3a"):
    tr = set(str(Path(p).resolve()) for p in out[f"{m}_train"]); va = set(str(Path(p).resolve()) for p in out[f"{m}_val"])
    check(not (tr & va), f"{m}: 학습·검증 겹침 0장"); check(not ((tr | va) & test), f"{m}: 학습·검증과 테스트 ②③ 겹침 0장")
    check(all(Path(p).exists() and lab(p).exists() for p in out[f"{m}_train"] + out[f"{m}_val"]), f"{m}: 그림·라벨 파일 모두 있음")
check(all(lab(p).read_text().strip() == "" for sp in neg for p in neg[sp]), "드론 아님 그림 모두 박스 없음")
Y = "path: " + str(ROOT / "data/derived") + "\ntrain: split_v10/{m}_train.txt\nval: split_v10/{m}_val.txt\nnames:\n{names}"
three = "  0: drone\n  1: wifi\n  2: bt\n"; one = "  0: drone\n"
for m, names in (("1a", three), ("2a", three), ("3a", one)): (ROOT / f"data/yaml/v17_{m}.yaml").write_text(Y.format(m="m" + m, names=names))
L.append("모두 통과" if ok else "실패가 있음 — 학습하지 말 것")
(ROOT / "runs/review/build_v17.txt").write_text("\n".join(L) + "\n"); print("\n".join(L))
if ok: print("BUILD_V17_DONE")
