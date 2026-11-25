"""실제 배경 모델: 혼잡한 장소의 실제 배경에 합성 드론 신호를 넣어 학습 (2026-11-15). 데이터 만들기·평가·결과 표·진행 상황. 원고 4.3절, 6.3절 표 13·14, 보충 자료 S7.
코드 이름과 원고 이름:
  test2_1 = 객차 배경 모델: 객차 드론 전원 꺼짐 IQ 데이터를 배경으로 학습 → 수원역으로 평가.
  test2_2 = 수원역 배경 모델: 수원역 드론·조종기 전원 꺼짐 IQ 데이터를 배경으로 학습 → 객차로 평가.
  비교 대상인 생성 간섭 모델(80 % 조건, 모델 10개)의 결과는 paper/src/rev1_eval.py·rev1_test1_4.py 가 만든 값을 읽는다.
두 모델 공통: 드론 한 클래스(0: drone, 와이파이·블루투스 박스 없음). 드론 신호 세그먼트 = 공개 20개 기종 303개 전부(Neo 2 없음, fold 나누지 않음),
  학습·검증 데이터 세그먼트는 rev1 과 같은 part 로 나눔. 이미지 확률은 rev1(v10 규칙)과 같다: 15 % 는 드론 없음, 나머지의 70 % 에 드론(그중 15 % 는 2개),
  SNR 6–25 dB, 대역폭의 50–100 % 가 보이게. train 6,000장, validation 600장, seed 1·2. 학습은 rev1 과 같다(YOLO26s, 40 epoch, batch 16, last.pt).
  threshold = 자기 validation 600장에서 이미지 단위 F1 이 가장 큰 값(rev1_eval.tune). 테스트 데이터로 고르지 않는다.
실제 배경: 공원 배경과 수신 설정(수신 필터 20 MHz)·int8 단위가 같으므로 공원 배경(Park02_1)의 RMS 로 나눠 수신기 잡음 전력을 1 에 맞춘다
  (주파수 구간별 하위 10 % 세기의 중앙값: 공원 1,285, 객차 1,377–1,569, 수원역 2,160–2,191. 계산은 paper/src/noise_level.py → runs/rev1/exp2/noise_level.json.
   수원역은 약한 신호가 늘 깔려 있어 높음 → 그대로 둠).
  실제 배경에는 와이파이·블루투스를 더하지 않는다(실제 신호가 이미 있다). 20 ms 창은 IQ 파일 안에서만, 파일 길이에 비례해 고른다(memmap 으로 읽음).
  드론 없는 이미지 = 실제 배경 그대로(정답 박스 없음).
세션 나누기(결과를 보기 전에 정한 규칙, 11-15 계획 표 2):
  test2_1 train = 객차 드론 끔 세션 8개 중 묶음(혼잡도 미기록, 한산한 객실)마다 시간상 마지막 세션을 뺀 6개(IQ 파일 12개), validation = 뺀 2개(IQ 파일 4개).
  test2_2 train = 수원역 드론 끔, 영상 켜기 전(IQ 파일 2개), validation = 수원역 드론 끔, 영상 끈 뒤(IQ 파일 2개).
  Neo 2 를 켠 IQ 파일은 어떤 모델의 학습·validation 에도 쓰지 않는다(build 의 누출 점검이 확인 → EXP2_LEAK_OK).
명령:
  build  : data/derived/rev1/<이름>/{images,labels,meta.json}, paper/runs/rev1/exp2/{lists,yaml,leak.txt} → EXP2_BUILD_DONE
  eval   : 모델마다 검증 데이터의 판정 임계값, 혼잡 장소 실측 이미지 15,163장 전부, 저간섭 실측 평가(baseline) → paper/runs/rev1/exp2/eval/<이름>/ → EXP2_EVAL_DONE
  report : paper/runs/rev1/exp2/RESULTS_exp2.md, results_exp2.json → EXP2_REPORT_DONE
  status : 진행 상황(체인이 매시 정각과 30분에 불러 paper/runs/rev1/exp2/status.log 에 적음)
사용: python paper/src/rev1_exp2.py build [--n-train N --n-val N --root 폴더 --only 이름 …] | eval [--only 이름 … --root 폴더 --weights 파일] | report | status"""
import sys, json, argparse, time, gc
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C, rev1_test1_4 as T
sys.path.insert(0, str(C.ROOT / "src"))
import synth_iq as S

X = C.RUNS / "exp2"; KINDS = ("test2_1", "test2_2"); NAMES = [f"{k}_s{r}" for r in C.REPS for k in KINDS]
SEEDX = {"train": 41, "val": 42}; KCODE = {"test2_1": 1, "test2_2": 2}
TITLE = {"test2_1": "객차 배경 모델", "test2_2": "수원역 배경 모델"}
TRAIN_VENUE = {"test2_1": "T", "test2_2": "S"}
VNAME = {"S": "수원역", "T": "객차"}
SHORT = T.GNAME                                                   # 취득 조건 이름(보충 자료 표 S13 과 같음)
MAIN = [("S", "rec_S_ground", ["S_on_fixed", "S_on_unconf"], True, "재현율: 수원역, 비비행 상태, 영상 전송"), ("S", "rec_S_flight", ["S_flight"], True, "재현율: 수원역, 비행 중"),
        ("S", "fpr_S", ["S_off_before", "S_off_after"], False, "FPR: 수원역, 드론 전원 꺼짐"),
        ("T", "rec_T_ground", ["T_on_unrec", "T_on_6m", "T_on_10m"], True, "재현율: 객차, 비비행 상태, 영상 전송"), ("T", "fpr_T", ["T_off_unrec", "T_off_quiet"], False, "FPR: 객차, 드론 전원 꺼짐")]   # 원고 표 13 의 열


# ---------------- 데이터 나누기와 누출 점검 ----------------
def off_files():
    """반환 {test2_1: {train: [...], val: [...]}, test2_2: {...}} — d4d 폴더 기준 상대 경로."""
    sess = {}
    for f in sorted(T.RAW.glob("02_열차_객실/드론_꺼짐/*/*/*/iq.sigmf-data")):
        sess.setdefault(f.parents[2].name, {}).setdefault(f.parents[1].name, []).append(f)       # 묶음 → 세션(이름 = 시작 시각) → IQ 파일
    last = {sub: max(s) for sub, s in sess.items()}                                              # 묶음마다 시간상 마지막 세션 = validation
    test2_1 = dict(train=[f for sub, s in sess.items() for k, fs in s.items() if k != last[sub] for f in fs], val=[f for sub, s in sess.items() for f in s[last[sub]]])
    sw = sorted(T.RAW.glob("01_수원역_플랫폼/드론_꺼짐/*/iq.sigmf-data"))
    test2_2 = dict(train=[f for f in sw if "_post_link_off_" not in str(f)], val=[f for f in sw if "_post_link_off_" in str(f)])
    rel = lambda L: [str(p.relative_to(C.ROOT)) for p in sorted(L)]
    return {"test2_1": {k: rel(v) for k, v in test2_1.items()}, "test2_2": {k: rel(v) for k, v in test2_2.items()}}


def leak_check(F):
    L = ["실험 2 누출 점검 (paper/src/rev1_exp2.py build)", ""]
    for k, d in F.items():
        assert not set(d["train"]) & set(d["val"]), k
        for role, files in d.items():
            gs = [T.group_of(str(Path(p).relative_to(T.RAW.relative_to(C.ROOT)))) for p in files]
            assert all(not T.GDRONE[g] for g in gs), (k, role, "Neo 2 를 켠 IQ 파일이 들어 있음")
            assert all(g[0] == TRAIN_VENUE[k] for g in gs), (k, role, "학습 장소가 아닌 IQ 파일")
            L.append(f"{k} {role}: IQ 파일 {len(files)}개, 묶음 {sorted(set(gs))}")
            L += [f"  {p}" for p in files]
    L += ["", "Neo 2 를 켠 IQ 파일 없음, 학습·검증 데이터 겹침 없음, 학습 장소 밖 IQ 파일 없음 → EXP2_LEAK_OK"]
    return L


def piece_split(meta):
    val = [i for i, m in enumerate(meta) if m["part"] == "val" and not m["group"].startswith("Tampere")]   # rev1_common.split_pieces 와 같은 기준
    sv = set(val); return dict(train=[i for i in range(len(meta)) if i not in sv], val=val)


# ---------------- 합성 ----------------
_W = {}


def _init():
    import rev1_synth as R
    R._init(); _W["R"] = R; _W["arr"], _W["meta"] = R._W["arr"], R._W["meta"]; _W["sd"] = R._W["train"][1]   # 공원 배경 Park02_1 의 RMS
    _W["mm"] = {}


def _mm(p):
    if p not in _W["mm"]: _W["mm"][p] = np.memmap(C.ROOT / p, dtype=np.int8, mode="r")
    return _W["mm"][p]


def compose_real(kind, role, pool, idx, rep, files):
    arr, meta = _W["arr"], _W["meta"]; key = [SEEDX[role], KCODE[kind], rep, idx]; rng = np.random.default_rng(key)
    empty = rng.random() < 0.15; nd = (2 if rng.random() < 0.15 else 1) if (not empty and rng.random() < 0.7) else 0
    lens = np.array([len(_mm(p)) // 2 for p in files]); fi = int(rng.choice(len(files), p=lens / lens.sum())); off = int(rng.integers(0, lens[fi] - S.N + 1))
    r = np.asarray(_mm(files[fi])[2 * off: 2 * (off + S.N)], dtype=np.float32); x = ((r[0::2] + 1j * r[1::2]) / _W["sd"]).astype(np.complex64)
    dboxes, used, taken = [], [], set()
    for _ in range(nd):
        j = int(rng.choice([q for q in pool if q not in taken])); taken.add(j)
        p = dict(meta[j]); p["iq"] = np.array(arr[j]); snr_t = float(rng.uniform(6, 25))
        fo = S.place_drone_v8(rng, p["bw"]); p["iq"] = S.cut_to_window(p["iq"], fo)
        x, b, snr = S.add_drone(x, p, rng, snr_t, fo=fo); dboxes.append(b)
        used.append(dict(piece=j, group=meta[j]["group"], snr=round(snr, 1), visible_mhz=round((b[4] - b[3]) / 1e6, 2), bw_mhz=round(p["bw"] / 1e6, 2)))
    return x, dboxes, dict(n_drone=nd, bg_file=files[fi], bg_offset=off, seed=key, pieces=used)


def _img_job(a):
    out, stem, kind, role, pool, idx, rep, extra = a
    x, boxes, info = compose_real(kind, role, pool, idx, rep, extra)
    _W["R"].write(Path(out), stem, x, boxes); info["stem"] = stem; return info


def build(n_train=C.N_TRAIN, n_val=C.N_VAL, root=None, only=None, workers=12):
    meta = C.pieces()[1]; PS = piece_split(meta); F = off_files(); L = leak_check(F)
    droot = Path(root) if root else C.DATA; x = Path(root) / "exp2" if root else X
    (x / "lists").mkdir(parents=True, exist_ok=True); (x / "yaml").mkdir(parents=True, exist_ok=True); (x / "leak.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:2] + [l for l in L if not l.startswith("  ")][2:]), flush=True)
    with Pool(workers, initializer=_init) as P:
        for name in NAMES:
            if only and name not in only: continue
            kind, rep = name.split("_s")[0], int(name.split("_s")[1]); d = droot / name
            if (d / "meta.json").exists(): print("있음, 건너뜀:", name, flush=True); continue
            (d / "images").mkdir(parents=True, exist_ok=True); (d / "labels").mkdir(parents=True, exist_ok=True); t = time.time(); js = []
            for role, n in (("train", n_train), ("val", n_val)):
                js += [(str(d), f"{role}_{i:05d}", kind, role, PS[role], i, rep, F[kind][role]) for i in range(n)]
            info = P.map(_img_job, js, chunksize=16)
            json.dump(info, open(d / "meta.json.tmp", "w")); (d / "meta.json.tmp").rename(d / "meta.json")
            nd = sum(r["n_drone"] > 0 for r in info); print(f"{name}: {len(info)}장 (드론 있음 {nd}장, 없음 {len(info) - nd}장), {time.time() - t:.0f}초", flush=True)
            for part in ("train", "val"):
                (x / "lists" / f"{name}_{part}.txt").write_text("".join(f"{d / 'images' / (r['stem'] + '.png')}\n" for r in info if r["stem"].startswith(part)))
            (x / "yaml" / f"{name}.yaml").write_text(f"train: ../lists/{name}_train.txt\nval: ../lists/{name}_val.txt\nnames:\n  0: drone\n")
    print("EXP2_LEAK_OK\nEXP2_BUILD_DONE", flush=True)


# ---------------- 평가 ----------------
def evaluate(only=None, root=None, weights=None):
    import torch
    from PIL import Image
    from ultralytics import YOLO
    import rev1_eval as E
    droot = Path(root) if root else C.DATA; x = Path(root) / "exp2" if root else X
    I = T.images(); G = None
    for name in NAMES:
        if only and name not in only: continue
        w = Path(weights) if weights else X / "models" / f"{name}.pt"; d = x / "eval" / name
        if not w.exists(): print("모델 없음, 건너뜀:", name, flush=True); continue
        if (d / "result.json").exists(): print("있음, 건너뜀:", name, flush=True); continue
        if G is None:
            t0 = time.time(); G = [np.array(Image.open(C.ROOT / p)) for p in I.image]; print(f"1-4 이미지 {len(G)}장 읽음 ({time.time() - t0:.0f}초)", flush=True)
        d.mkdir(parents=True, exist_ok=True); t1 = time.time(); m = YOLO(str(w))
        val = sorted(str(p) for p in (droot / name / "images").glob("val_*.png"))
        V = pd.DataFrame(E.syn_rows(m, val, set="val")); V.to_csv(d / "val.csv", index=False); th = E.tune(V); t = th["threshold"]
        rows = []
        for (_, r), boxes in zip(I.iterrows(), E.predict(m, G)):
            hit = [b[4] for b in boxes if E.iou1(b[1], b[3], *E.CH) >= E.IOU]
            rows.append(dict(clip=r["clip"], group=r["group"], frame=int(r.frame), max_conf=max([b[4] for b in boxes], default=0.0), max_hit=max(hit, default=0.0)))
        T3 = pd.DataFrame(rows); T3.to_csv(d / "test1_4.csv", index=False)
        B = E.baseline(m); B.to_csv(d / "test1_3.csv", index=False)
        res = dict(name=name, weights=str(w.relative_to(C.ROOT)) if w.is_absolute() and C.ROOT in w.parents else str(w), val=th, test3={},
                   baseline_recall={s: dict(all=float((q.max_hit >= t).mean()), n=len(q)) for s, q in B[B.role != "fpr"].groupby("session")},
                   baseline_fpr={s: dict(fpr=float((q.max_conf >= t).mean()), n=len(q)) for s, q in B[B.role == "fpr"].groupby("session")})
        for g, _, dr in T.GROUPS:
            q = T3[T3.group == g]
            res["test3"][g] = dict(n=len(q), rate=float(((q.max_hit if dr else q.max_conf) >= t).mean()),
                                   clips={c: float(((qq.max_hit if dr else qq.max_conf) >= t).mean()) for c, qq in q.groupby("clip")})
        json.dump(res, open(d / "result.json", "w"), ensure_ascii=False, indent=1)
        print(name, f"threshold {t}, validation F1 {th['f1']}", {SHORT[g]: round(v["rate"] * 100, 1) for g, v in res["test3"].items()}, f"{time.time() - t1:.0f}초", flush=True)
        del m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("EXP2_EVAL_DONE", flush=True)


# ---------------- 결과 표 ----------------
def _old(train="c80"):
    """생성 간섭 모델 중 한 학습 조건(모델 10개)의 혼잡 장소 실측 평가 결과: 모델마다 (fold, result, test1_4.csv)."""
    out = []
    for f in sorted((C.RUNS / "eval").glob(f"*_{train}_s*/test1_4.json")):
        out.append(dict(name=f.parent.name, fold=f.parent.name.rsplit(f"_{train}", 1)[0], t3=json.load(open(f)), res=json.load(open(f.parent / "result.json")),
                        csv=pd.read_csv(f.parent / "test1_4.csv")))
    return out


def _rate(df, groups, drone, t):
    q = df[df.group.isin(groups)]; return float(((q.max_hit if drone else q.max_conf) >= t).mean())


def _auc_v(df, v):
    p = df[df.group.isin([g for g, _, d in T.GROUPS if d and g[0] == v])].max_hit.to_numpy()
    n = df[df.group.isin([g for g, _, d in T.GROUPS if not d and g[0] == v])].max_conf.to_numpy(); return T.auc(p, n)


def _rec_at(df, v, fpr=0.10):
    p = df[df.group.isin([g for g, _, d in T.GROUPS if d and g[0] == v])].max_hit.to_numpy()
    n = df[df.group.isin([g for g, _, d in T.GROUPS if not d and g[0] == v])].max_conf.to_numpy()
    ts = np.round(np.arange(0.01, 0.951, 0.01), 2); fx = np.array([(n >= t).mean() for t in ts]); ry = np.array([(p >= t).mean() for t in ts]); o = np.argsort(fx)
    return float(np.interp(fpr, fx[o], ry[o]))


def report():
    M = {}
    for name in NAMES:
        f = X / "eval" / name / "result.json"
        if f.exists(): M[name] = dict(res=json.load(open(f)), csv=pd.read_csv(f.parent / "test1_4.csv"), base=pd.read_csv(f.parent / "test1_3.csv"))
    olds = _old("c80"); OLD = "생성 간섭 모델, 80 %"
    fmt = lambda v: "" if v is None else f"{v * 100:.1f}"
    L = ["# 실제 배경 모델 결과", "",
         "paper/src/rev1_exp2.py report 가 만든 표. 모델 정의와 세션 나누기는 그 파일 머리말, 데이터 누출 점검은 leak.txt.",
         "값은 시드 2개(1, 2)의 평균이다(모델이 하나만 끝났으면 그 값). '생성 간섭 모델, 80 %'는 80 % 조건 모델 10개(fold 5개 × 시드 2개)의 fold 균형 평균이다.",
         "학습이나 검증에 쓴 취득 조건은 '학습에 사용'으로 적고 값을 내지 않는다.", "", f"평가가 끝난 모델: {', '.join(M) or '없음'}", ""]
    L += ["**표 1. 모델별 판정 임계값과 검증 데이터의 이미지 단위 F1 점수 (단위 없음, 검증 이미지 600장)**", "", "| 모델 | 시드 | 판정 임계값 | F1 점수 | 드론 존재 이미지 | 드론 부재 이미지 |", "|---|---:|---:|---:|---:|---:|"]
    for n, r in M.items(): v = r["res"]["val"]; L.append(f"| {TITLE[n.split('_s')[0]]} | {n[-1]} | {v['threshold']:.2f} | {v['f1']:.3f} | {v['n_pos']} | {v['n_neg']} |")
    kinds = [k for k in KINDS if any(n.startswith(k) for n in M)]
    def kmean(k, fn):
        vals = [fn(r) for n, r in M.items() if n.startswith(k)]; vals = [v for v in vals if v is not None]; return float(np.mean(vals)) if vals else None
    def omean(rows, fn): return float(np.mean([np.mean([fn(r) for r in rows if r["fold"] == fk]) for fk in sorted({r["fold"] for r in rows})])) if rows else None
    L += ["", "**표 2. 취득 조건을 합친 재현율과 FPR (단위: %, 검증 데이터에서 정한 판정 임계값으로 판정, 원고 표 13)**", "", "| 모델 | " + " | ".join(t for *_, t in MAIN) + " |", "|---|" + "---:|" * len(MAIN)]
    for k in kinds:
        cells = []
        for v, key, gs, dr, _ in MAIN:
            if not dr and TRAIN_VENUE[k] == v: cells.append("학습에 사용"); continue
            val = kmean(k, lambda r: _rate(r["csv"], gs, dr, r["res"]["val"]["threshold"])); s = fmt(val)
            cells.append(s + (" (참고)" if dr and TRAIN_VENUE[k] == v else ""))
        L.append(f"| {TITLE[k]} | " + " | ".join(cells) + " |")
    L.append(f"| {OLD} | " + " | ".join(fmt(omean(olds, lambda r, gs=gs, dr=dr: _rate(r["csv"], gs, dr, r["res"]["val"]["threshold"]))) for v, key, gs, dr, _ in MAIN) + " |")
    L += ["", "'(참고)'는 배경 장소가 학습과 같은 재현율이다(Neo 2 신호는 학습에 쓰지 않음).", "",
          "**표 3. 장소별 AUC (단위 없음, 0.5 = 구분 능력 없음, 1 = 완전한 구분, 원고 표 14)**", "", "| 모델 | 수원역 대합실 | 전철 1호선 객차 |", "|---|---:|---:|"]
    for k in kinds:
        L.append(f"| {TITLE[k]} | " + " | ".join("학습에 사용" if TRAIN_VENUE[k] == v else f"{kmean(k, lambda r, v=v: _auc_v(r['csv'], v)):.3f}" for v in "ST") + " |")
    L.append(f"| {OLD} | " + " | ".join(f"{omean(olds, lambda r, v=v: _auc_v(r['csv'], v)):.3f}" for v in "ST") + " |")
    L += ["", "**표 4. 판정 임계값을 바꿔 FPR을 10 %로 맞췄을 때의 재현율 (단위: %, 평가 데이터의 곡선에서 읽은 참고값)**", "", "| 모델 | 수원역 대합실 | 전철 1호선 객차 |", "|---|---:|---:|"]
    for k in kinds:
        L.append(f"| {TITLE[k]} | " + " | ".join("학습에 사용" if TRAIN_VENUE[k] == v else fmt(kmean(k, lambda r, v=v: _rec_at(r['csv'], v))) for v in "ST") + " |")
    L.append(f"| {OLD} | " + " | ".join(fmt(omean(olds, lambda r, v=v: _rec_at(r['csv'], v))) for v in "ST") + " |")
    # bootstrap (IQ 파일 단위, 이미지 수로 무게, 같은 표본을 seed 두 모델에 함께)
    L += ["", "**표 5. 표 2의 IQ 파일 단위 부트스트랩 95 % 신뢰구간 (단위: %, 2,000회)**", "", "| 모델 | " + " | ".join(t for *_, t in MAIN) + " |", "|---|" + "---:|" * len(MAIN)]
    rng = np.random.default_rng(0)
    for k in kinds:
        rs = [r for n, r in M.items() if n.startswith(k)]; cells = []
        for v, key, gs, dr, _ in MAIN:
            if not dr and TRAIN_VENUE[k] == v: cells.append("학습에 사용"); continue
            per = []
            for r in rs:
                q = r["csv"][r["csv"].group.isin(gs)]; t = r["res"]["val"]["threshold"]
                per.append(q.groupby("clip").apply(lambda z: pd.Series(dict(n=len(z), k=int(((z.max_hit if dr else z.max_conf) >= t).sum())))))
            clips = per[0].index.to_numpy(); nimg = per[0].n.to_numpy(float); Kc = np.stack([p.loc[clips].k.to_numpy(float) for p in per])
            idx = rng.integers(0, len(clips), (2000, len(clips))); est = (Kc[:, idx].sum(2) / nimg[idx].sum(1)).mean(0)
            cells.append(f"{np.percentile(est, 2.5) * 100:.1f}–{np.percentile(est, 97.5) * 100:.1f}")
        L.append(f"| {TITLE[k]} | " + " | ".join(cells) + " |")
    # 저간섭 실측 평가(baseline)
    rec_s = ["01", "03", "09", "10", "13", "07", "indoor_ON"]; fpr_s = ["02", "indoor_OFF"]
    SN = {"01": "공원 세션 01 (조종기와 1 m)", "03": "공원 세션 03 (조종기와 10 m)", "09": "공원 세션 09 (고도 10 m)", "10": "공원 세션 10 (고도 10 m)",
          "13": "공원 세션 13 (근처 자유 비행)", "07": "공원 세션 07 (고도 30 m)", "indoor_ON": "실내, Neo 2 전원 켜짐",
          "02": "공원, 드론·조종기 전원 꺼짐 (세션 02)", "indoor_OFF": "실내, Neo 2 전원 꺼짐"}          # 보충 자료 표 S25·S26 의 열
    def bmean(k, s, kind): return kmean(k, lambda r: r["res"]["baseline_recall"][s]["all"] if kind == "rec" else r["res"]["baseline_fpr"][s]["fpr"])
    def bold(rows, s, kind): return omean(rows, lambda r: r["res"]["baseline_recall"][s]["all"] if kind == "rec" else r["res"]["baseline_fpr"][s]["fpr"])
    for kind, ss, title in (("rec", rec_s, "**표 6. 저간섭 실측 평가의 재현율 (단위: %, 세션별)**"), ("fpr", fpr_s, "**표 7. 저간섭 실측 평가의 FPR (단위: %, 세션별)**")):
        L += ["", title, "", "| 모델 | " + " | ".join(SN[s] for s in ss) + " |", "|---|" + "---:|" * len(ss)]
        for k in kinds: L.append(f"| {TITLE[k]} | " + " | ".join(fmt(bmean(k, s, kind)) for s in ss) + " |")
        L.append(f"| {OLD} | " + " | ".join(fmt(bold(olds, s, kind)) for s in ss) + " |")
    # 모델별, 묶음별
    L += ["", "**표 8. 모델과 취득 조건별 재현율(드론 존재)과 FPR(드론 전원 꺼짐) (단위: %)**", "", "| 모델 | 시드 | " + " | ".join(SHORT[g] for g, _, _ in T.GROUPS) + " |", "|---|---:|" + "---:|" * len(T.GROUPS)]
    for n, r in M.items():
        k = n.split("_s")[0]
        L.append(f"| {TITLE[k]} | {n[-1]} | " + " | ".join("학습에 사용" if (not d and TRAIN_VENUE[k] == g[0]) else fmt(r["res"]["test3"][g]["rate"]) for g, _, d in T.GROUPS) + " |")
    (X / "RESULTS_exp2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    json.dump({n: r["res"] for n, r in M.items()}, open(X / "results_exp2.json", "w"), ensure_ascii=False, indent=1)
    print("\n".join(L)); print("EXP2_REPORT_DONE", flush=True)


# ---------------- 진행 상황 ----------------
def status():
    s = [time.strftime("%m-%d %H:%M")]
    nb = sum((C.DATA / n / "meta.json").exists() for n in NAMES); s.append(f"합성 {nb}/{len(NAMES)}벌")
    for n in NAMES:
        k = n.split("_s")[0]; e = X / "eval" / n / "result.json"; rc = X / f"det_{n}" / "train" / "results.csv"
        if e.exists():
            r = json.load(open(e)); t3 = r["test3"]; v = "S" if k == "test2_1" else "T"
            gs_on = [g for g, _, d in T.GROUPS if d and g[0] == v]; gs_off = [g for g, _, d in T.GROUPS if not d and g[0] == v]
            rec = sum(t3[g]["rate"] * t3[g]["n"] for g in gs_on) / sum(t3[g]["n"] for g in gs_on); fpr = sum(t3[g]["rate"] * t3[g]["n"] for g in gs_off) / sum(t3[g]["n"] for g in gs_off)
            s.append(f"{n}: 평가 끝(threshold {r['val']['threshold']:.2f}, {VNAME[v]} recall {rec * 100:.1f} %, FPR {fpr * 100:.1f} %)")
        elif (X / "models" / f"{n}.pt").exists(): s.append(f"{n}: 학습 끝, 평가 중")
        elif rc.exists(): s.append(f"{n}: 학습 중 {max(0, len(open(rc).read().strip().splitlines()) - 1)}/40 epoch")
        elif (X / f"det_{n}").exists(): s.append(f"{n}: 학습 시작")
        else: s.append(f"{n}: 대기")
    print(" | ".join(s), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "eval", "report", "status"])
    ap.add_argument("--n-train", type=int, default=C.N_TRAIN); ap.add_argument("--n-val", type=int, default=C.N_VAL); ap.add_argument("--root"); ap.add_argument("--only", nargs="*")
    ap.add_argument("--weights"); ap.add_argument("--workers", type=int, default=12); A = ap.parse_args()
    if A.cmd == "build": build(A.n_train, A.n_val, A.root, A.only, A.workers)
    elif A.cmd == "eval": evaluate(A.only, A.root, A.weights)
    elif A.cmd == "report": report()
    else: status()
