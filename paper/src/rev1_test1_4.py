"""혼잡 장소 실측 평가(코드 이름 test1_4): 실제 혼잡 장소의 Neo 2 IQ 데이터로 생성 간섭 모델 40개를 평가 (2026-11-15). 원고 6.2절 표 11·12, 보충 자료 S6.
데이터: data/raw/hackrf/4_neo2_crowded_20261114 (2026-11-14 취득, 압축 원본은 data/HackRF/4_neo2_crowded_20261114/)
  01_수원역_플랫폼(수원역 지하 1층 대합실, 약 300명, 드론·조종기는 SDR 바로 옆), 02_열차_객실(수도권 전철 1호선 객차 안).
  수신 설정은 공원(3_neo2_park)과 같다: HackRF One + SDRangel, 중심 2,449.5 MHz, 20 MS/s, 수신 필터 20 MHz, LNA 16 + VGA 16 dB, 증폭기 끔.
  Neo 2 채널은 2,444.5–2,454.5 MHz(10 MHz, 수동 고정). 수원역의 "채널 확인 전" 클립 2개만 송신 채널을 취득 때 확인하지 않았다.
이미지: data/derived/hackrf/4_neo2_crowded_20261114/<장소>/images (src/iq_to_image.py, 학습과 같은 변환, 20 ms 한 장)
정답: 수집자가 적은 클립 단위 장비 상태. 영상 켜짐·비행 클립의 모든 이미지 = 드론 있음, 드론 꺼짐 클립의 모든 이미지 = 드론 없음.
  이미지마다의 정답은 없다. Neo 2 는 영상 링크가 켜져 있으면 2.45 ms 또는 4.8 ms 마다 버스트를 보내므로 20 ms 이미지마다 버스트가 들어간다
  (clips 명령의 버스트 측정으로 클립마다 확인한다).
판정: paper/src/rev1_eval.py 의 baseline 과 같다. 드론 있는 이미지 = 드론 박스의 주파수 범위가 Neo 2 채널(이미지 세로 0.25–0.75)과 IoU 0.3 이상이고
  confidence 가 threshold 이상이면 정탐. 드론 없는 이미지 = 드론 박스(위치 무관)가 하나라도 threshold 이상이면 오탐.
  threshold 는 모델마다 자기 validation 에서 정한 값(paper/runs/rev1/eval/<이름>/result.json). 혼잡 장소 실측 데이터는 학습·threshold·모델 선택에 쓰지 않는다.
명령:
  clips   클립 목록과 클립별 버스트 길이·반복 간격(paper/src/neo2_burst_sessions.py 와 같은 방법) → paper/runs/rev1/test1_4/clips.csv
  eval    생성 간섭 모델 40개(이미지는 한 번만 읽어 메모리에 둠) → paper/runs/rev1/eval/<이름>/test1_4.csv(이미지마다 max_conf·max_hit), test1_4.json
          이미 test1_4.json 이 있는 모델은 건너뛴다. 끝나면 REV1_TEST3_DONE
  report  표 → paper/runs/rev1/test1_4/RESULTS_test1_4.md, results_test1_4.json
사용: python paper/src/rev1_test1_4.py clips | eval | report"""
import json, sys, gc, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rev1_common as C

RAW = C.ROOT / "data/raw/hackrf/4_neo2_crowded_20261114"; DER = C.ROOT / "data/derived/hackrf/4_neo2_crowded_20261114"
OUT = C.RUNS / "test1_4"; FC = 2449.5e6
VENUES = ["01_수원역_플랫폼", "02_열차_객실"]
# (묶음, 표에 쓰는 이름, 드론 있음)
GROUPS = [("S_on_fixed", "수원역, 비비행 상태 영상 전송 (채널 고정 확인)", True), ("S_on_unconf", "수원역, 비비행 상태 영상 전송 (채널 확인 전)", True),
          ("S_flight", "수원역, 비행 중", True),
          ("T_on_unrec", "객차, 비비행 상태 영상 전송 (거리 기록 없음)", True), ("T_on_6m", "객차, 비비행 상태 영상 전송 (6 m)", True), ("T_on_10m", "객차, 비비행 상태 영상 전송 (10 m)", True),
          ("S_off_before", "수원역, 드론·조종기 전원 꺼짐 (영상 전송 전)", False), ("S_off_after", "수원역, 드론·조종기 전원 꺼짐 (영상 전송 후)", False),
          ("T_off_unrec", "객차, 드론 전원 꺼짐 (혼잡도 기록 없음)", False), ("T_off_quiet", "객차, 드론 전원 꺼짐 (한산한 객실)", False)]   # 이름은 보충 자료 표 S13 과 같다
GNAME = {g: n for g, n, _ in GROUPS}; GDRONE = {g: d for g, _, d in GROUPS}
# 원고 표 11 의 열: (이름, 합치는 묶음, 드론 있음)
MERGED = [("rec_S_ground", ["S_on_fixed", "S_on_unconf"], True), ("rec_S_flight", ["S_flight"], True), ("rec_T_ground", ["T_on_unrec", "T_on_6m", "T_on_10m"], True),
          ("fpr_S", ["S_off_before", "S_off_after"], False), ("fpr_T", ["T_off_unrec", "T_off_quiet"], False)]
MERGED_NAMES = [("rec_S_ground", "재현율: 수원역, 비비행 상태, 영상 전송 (짧은 버스트)"), ("rec_S_flight", "재현율: 수원역, 비행 중 (긴 버스트)"),
                ("rec_T_ground", "재현율: 객차, 비비행 상태, 영상 전송 (짧은 버스트)"), ("fpr_S", "FPR: 수원역, 드론 전원 꺼짐"), ("fpr_T", "FPR: 객차, 드론 전원 꺼짐")]


def short(c):
    """클립 폴더 이름(예: 20261114T042331Z_neo2_off_01_f060d2ba) → 표에 쓰는 이름(취득 시작 한국 시각 + 라벨, 예: 13:23:31 off_01)."""
    return f"{int(c[9:11]) + 9:02d}:{c[11:13]}:{c[13:15]} {c.split('_neo2_')[1][:-9]}"


def group_of(rel):
    """data/raw/hackrf/4_neo2_crowded_20261114 아래 상대 경로(문자열) → 묶음."""
    if rel.startswith("01_수원역_플랫폼/"):
        if "/드론_비행_중/" in rel: return "S_flight"
        if "/드론_영상_켜짐/" in rel: return "S_on_fixed" if "_on_fixed10_" in rel else "S_on_unconf"
        if "/드론_꺼짐/" in rel: return "S_off_after" if "_post_link_off_" in rel else "S_off_before"
    else:
        for k, g in (("/드론_영상_켜짐/거리_미기록/", "T_on_unrec"), ("/드론_영상_켜짐/6m/", "T_on_6m"), ("/드론_영상_켜짐/10m/", "T_on_10m"),
                     ("/드론_꺼짐/혼잡도_미기록/", "T_off_unrec"), ("/드론_꺼짐/한산한_객실/", "T_off_quiet")):
            if k in rel: return g
    raise ValueError(rel)


def images():
    """이미지 목록: image(경로, d4d 폴더 기준), clip(클립 폴더 이름), group, drone, frame. 묶음 순서 → 클립 → 프레임."""
    D = []
    for v in VENUES:
        x = pd.read_csv(DER / v / "images/index.csv")
        x["image"] = [str((DER / v / "images" / p).relative_to(C.ROOT)) for p in x.image]
        x["clip"] = [Path(s).parent.name for s in x.source]; x["group"] = [group_of(str(Path(s).relative_to(RAW.relative_to(C.ROOT)))) for s in x.source]
        D.append(x)
    D = pd.concat(D, ignore_index=True); D["drone"] = D.group.map(GDRONE)
    D["g_ord"] = D.group.map({g: i for i, (g, _, _) in enumerate(GROUPS)})
    return D.sort_values(["g_ord", "clip", "frame"]).drop(columns="g_ord").reset_index(drop=True)


def _burst(a):
    import neo2_burst_sessions as B
    clip, f = a; return B.session(clip, [(f, FC)], FC)


def _occ(a):
    """클립의 20 ms 마다 스펙트럼 점유율(paper/src/occupancy.py 의 계산: 97 × 512 칸, 플로어보다 6 dB 이상 센 칸의 비율, DC ±100 kHz 뺌).
    플로어 두 가지: 클립 = 그 클립 전체에서 줄마다 하위 10 %(저간섭 실측 평가의 점유율과 같음), 공원 = 공원 배경 0–1 s(Park02_1, 같은 수신 설정)에서 같은 값.
    전체 = 모든 줄, 채널 밖 = Neo 2 채널(가운데 ±5 MHz)에 여유 1.2배를 둔 ±6 MHz 줄을 뺌."""
    import occupancy as E
    clip, f, pf = a; Ps = np.stack([E.power(x) for x in E.read_iq(f)]); cf = np.percentile(Ps.reshape(-1, E.NF), 10, axis=0)
    ch = np.broadcast_to((np.abs(E.FR) <= 6e6)[None, :], Ps.shape[1:])
    return [dict(clip=clip, frame=i, occ_all=E.occ(P, cf)[0], occ_out=E.occ(P, cf, ch)[0], occ_all_park=E.occ(P, pf)[0], occ_out_park=E.occ(P, pf, ch)[0])
            for i, P in enumerate(Ps)]


def park_floor():
    import occupancy as E
    x = np.load(C.ROOT / "data/derived/hackrf_clean/Park02_1.npy"); Ps = np.stack([E.power(x[i:i + E.N]) for i in range(0, len(x) - E.N + 1, E.N)])
    return np.percentile(Ps.reshape(-1, E.NF), 10, axis=0)


def clips():
    OUT.mkdir(parents=True, exist_ok=True); I = images()
    files = sorted(RAW.rglob("iq.sigmf-data")); rows = []
    for f in files:
        meta = json.load(open(f.with_suffix(".sigmf-meta"), encoding="utf-8-sig")); n = f.stat().st_size // 2
        assert meta["captures"][0]["core:frequency"] == FC and meta["global"]["core:sample_rate"] == 20e6, f
        rel = str(f.relative_to(RAW)); clip = f.parent.name
        rows.append(dict(clip=clip, venue=rel.split("/")[0], group=group_of(rel), drone=GDRONE[group_of(rel)], seconds=round(n / 20e6, 3),
                         n_images=int((I["clip"] == clip).sum()), start_utc=meta["captures"][0]["core:datetime"], path=str(f.relative_to(C.ROOT))))
    with Pool(6) as p: B = p.map(_burst, [(r["clip"], str(C.ROOT / r["path"])) for r in rows])
    for r, b in zip(rows, B): r.update({k: b[k] for k in ("n_segments", "n_measured", "n_bursts", "on_ms_median", "on_ms_q25", "on_ms_q75", "period_ms_median", "on_off_db_median")})
    pf = park_floor()
    with Pool(6) as p: O = pd.DataFrame([x for q in p.map(_occ, [(r["clip"], str(C.ROOT / r["path"]), pf) for r in rows]) for x in q])
    O.to_csv(OUT / "occ_frames.csv", index=False)
    for r in rows:
        q = O[O["clip"] == r["clip"]]
        for k in ("occ_all", "occ_out", "occ_all_park", "occ_out_park"): r[k + "_median"] = float(q[k].median())
    D = pd.DataFrame(rows); D["g_ord"] = D.group.map({g: i for i, (g, _, _) in enumerate(GROUPS)})
    D = D.sort_values(["g_ord", "clip"]).drop(columns="g_ord"); D.to_csv(OUT / "clips.csv", index=False)
    print(D.drop(columns=["path", "start_utc"]).to_string()); print("REV1_TEST3_CLIPS_DONE")


def evaluate():
    import torch
    from PIL import Image
    from ultralytics import YOLO
    import rev1_eval as E
    I = images(); t0 = time.time()
    G = [np.array(Image.open(C.ROOT / p)) for p in I.image]; print(f"이미지 {len(G)}장 읽음 ({time.time() - t0:.0f} 초)", flush=True)
    for w in sorted((C.RUNS / "models").glob("*.pt")):
        name = w.stem; d = C.RUNS / "eval" / name
        if (d / "test1_4.json").exists(): print("건너뜀", name, flush=True); continue
        t = json.load(open(d / "result.json"))["val"]["threshold"]; t1 = time.time(); m = YOLO(str(w)); rows = []
        for (_, r), boxes in zip(I.iterrows(), E.predict(m, G)):
            hit = [b[4] for b in boxes if E.iou1(b[1], b[3], *E.CH) >= E.IOU]
            rows.append(dict(clip=r["clip"], group=r["group"], frame=int(r.frame), max_conf=max([b[4] for b in boxes], default=0.0), max_hit=max(hit, default=0.0)))
        T = pd.DataFrame(rows); T.to_csv(d / "test1_4.csv", index=False)
        res = dict(name=name, threshold=t, groups={})
        for g, _, dr in GROUPS:
            q = T[T.group == g]
            res["groups"][g] = dict(n=len(q), rate=float(((q.max_hit if dr else q.max_conf) >= t).mean()), any_box=float((q.max_conf >= t).mean()),
                                   clips={c: float(((qq.max_hit if dr else qq.max_conf) >= t).mean()) for c, qq in q.groupby("clip")})
        json.dump(res, open(d / "test1_4.json", "w"), ensure_ascii=False, indent=1)
        print(name, f"t={t}", {g: round(v["rate"], 3) for g, v in res["groups"].items()}, f"{time.time() - t1:.0f} 초", flush=True)
        del m; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("REV1_TEST3_DONE", flush=True)


def auc(p, n):
    """P(드론 있는 이미지 점수 > 드론 없는 이미지 점수), 같으면 반(Mann-Whitney U 를 넓이로 바꾼 값)."""
    rk = pd.Series(np.r_[p, n]).rank().to_numpy(); return float((rk[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def report():
    import re
    import rev1_report as RP
    R = []
    for f in sorted((C.RUNS / "eval").glob("*/test1_4.json")):
        m = re.match(r"^(f\d_[a-z]+)_(c02|c20|c50|c80)_s(\d)$", f.parent.name)
        if m: R.append(dict(name=f.parent.name, fold=m[1], train=m[2], seed=int(m[3]), t3=json.load(open(f))))
    K = pd.read_csv(OUT / "clips.csv"); made = time.strftime("%Y-%m-%d %H:%M"); J = dict(made=made, models=[r["name"] for r in R])
    pos = [(g, n) for g, n, d in GROUPS if d]; neg = [(g, n) for g, n, d in GROUPS if not d]
    L = ["# 혼잡 장소 실측 평가 결과 (생성 간섭 모델)", "",
         f"paper/src/rev1_test1_4.py report 가 {made} 에 만든 표. 평가가 끝난 생성 간섭 모델 {len(R)}개. 데이터는 2026-11-14 취득(수원역 지하 1층 대합실, 수도권 전철 1호선 객차, 수신 필터 20 MHz).",
         "정답은 취득자가 적은 IQ 파일 단위 장비 상태다(영상 전송·비행 IQ 파일의 이미지 = 드론 존재, 드론 전원 꺼짐 IQ 파일 = 드론 부재). 이미지 한 장 = 20 ms.",
         "재현율 = 드론 존재 이미지 중 Neo 2 채널(2,444.5–2,454.5 MHz)에 판정 임계값 이상인 드론 경계 상자가 있는 비율. FPR = 드론 부재 이미지 중 판정 임계값 이상인 드론 경계 상자가 하나라도 있는 비율.",
         "값은 fold 마다 시드 평균을 낸 뒤 fold 를 같은 무게로 평균한 값(fold 균형 평균). 판정 임계값은 모델마다 자기 검증 데이터에서 정한 값.", "",
         "## 1. 데이터", "", "**표 1. 취득 조건별 데이터 양 (IQ 파일: 개, 길이: 초, 이미지: 장)**", "",
         "| 취득 조건 | 드론 | IQ 파일 | 길이 | 이미지 |", "|---|---|---:|---:|---:|"]
    for g, n, d in GROUPS:
        q = K[K.group == g]; L.append(f"| {n} | {'존재' if d else '부재'} | {len(q)} | {q.seconds.sum():.1f} | {q.n_images.sum()} |")
    L.append(f"| 합계 | | {len(K)} | {K.seconds.sum():.1f} | {K.n_images.sum()} |")
    O = pd.read_csv(OUT / "occ_frames.csv").merge(K[["clip", "group"]], on="clip"); J["occupancy"] = {}
    L += ["", "**표 1-2. 취득 조건별 스펙트럼 점유율 (단위: %, 20 ms 마다 잰 값의 중앙값, 괄호는 25–75 백분위수)**", "",
          "97 × 512 셀에서 잡음 세기보다 6 dB 이상 센 셀의 비율(DC ±100 kHz 뺌, 저간섭 실측 평가의 점유율과 같은 계산). "
          "IQ 파일 기준 = 그 IQ 파일 전체에서 주파수 구간마다 하위 10 % 값을 잡음 세기로 씀, 공원 기준 = 공원 배경 0–1 s 에서 같은 값(같은 수신 설정). 채널 밖 = Neo 2 채널 ±6 MHz 구간을 뺌.", "",
          "| 취득 조건 | 전체 (IQ 파일 기준) | 채널 밖 (IQ 파일 기준) | 전체 (공원 기준) | 채널 밖 (공원 기준) |", "|---|---:|---:|---:|---:|"]
    qs = lambda v: f"{100 * v.median():.1f} ({100 * v.quantile(0.25):.1f}–{100 * v.quantile(0.75):.1f})"
    for g, n, _ in GROUPS:
        q = O[O.group == g]; J["occupancy"][g] = {k: float(q[k].median()) for k in ("occ_all", "occ_out", "occ_all_park", "occ_out_park")}
        L.append(f"| {n} | {qs(q.occ_all)} | {qs(q.occ_out)} | {qs(q.occ_all_park)} | {qs(q.occ_out_park)} |")
    L += ["", "**표 2. IQ 파일별 Neo 2 채널의 버스트 (길이·반복 간격: ms, 가운데 ±5 MHz, 100 ms 마다 20 ms 구간을 잼, 중앙값)**", "",
          "짧은 버스트 = 길이 약 0.45 ms·간격 약 2.45 ms(공원 01, 실내 Neo 2 전원 켜짐), 긴 버스트 = 길이 약 2.8 ms·간격 약 4.8 ms(공원 비행 세션). 드론 전원 꺼짐 IQ 파일의 값은 간섭 신호를 잰 것이다.", "",
          "| 취득 조건 | IQ 파일 | 버스트 길이 | 길이 25 백분위수 | 길이 75 백분위수 | 반복 간격 |", "|---|---|---:|---:|---:|---:|"]
    f2 = lambda v: "" if pd.isna(v) else f"{v:.2f}"
    for _, r in K.iterrows(): L.append(f"| {GNAME[r.group]} | {short(r['clip'])} | {f2(r.on_ms_median)} | {f2(r.on_ms_q25)} | {f2(r.on_ms_q75)} | {f2(r.period_ms_median)} |")
    L += ["", "**표 3. IQ 파일별 버스트 측정 가능 구간 (구간: 개, 비율: %)**", "", "켜짐·꺼짐 세기 차이(상위 10 % ÷ 하위 10 %)가 1 dB 미만인 20 ms 구간은 잴 수 없음으로 둔다.", "",
          "| 취득 조건 | IQ 파일 | 잰 구간 | 잴 수 있던 구간 | 비율 |", "|---|---|---:|---:|---:|"]
    for _, r in K.iterrows(): L.append(f"| {GNAME[r.group]} | {short(r['clip'])} | {r.n_segments} | {r.n_measured} | {100 * r.n_measured / max(r.n_segments, 1):.0f} |")
    L += ["", "**표 4. IQ 파일별 켜짐·꺼짐 세기 차이 (단위: dB, 구간 중앙값)**", "", "| 취득 조건 | IQ 파일 | 세기 차이 |", "|---|---|---:|"]
    for _, r in K.iterrows(): L.append(f"| {GNAME[r.group]} | {short(r['clip'])} | {r.on_off_db_median:.1f} |")
    get = lambda r, g: r["t3"]["groups"][g]["rate"]
    L += ["", "## 2. 재현율과 FPR", ""]
    L += RP.table("**표 5. 드론 존재 취득 조건의 재현율 (단위: %)**", R, get, pos)
    L += RP.table("**표 6. 드론 존재 취득 조건의 재현율 범위 (단위: %, 모델 중 가장 낮은 값과 가장 높은 값)**", R, get, pos, rng=True)
    L += RP.table("**표 7. 드론 전원 꺼짐 취득 조건의 FPR (단위: %)**", R, get, neg)
    L += RP.table("**표 8. 드론 전원 꺼짐 취득 조건의 FPR 범위 (단위: %, 모델 중 가장 낮은 값과 가장 높은 값)**", R, get, neg, rng=True)
    for r in R:                                                             # 장소별 AUC: 드론 있는 이미지의 max_hit 대 드론 없는 이미지의 max_conf
        T3 = pd.read_csv(C.RUNS / "eval" / r["name"] / "test1_4.csv"); r["auc"] = {}
        for v in ("S", "T"):
            p = T3[T3.group.isin([g for g, _, d in GROUPS if d and g[0] == v])].max_hit.to_numpy(); n = T3[T3.group.isin([g for g, _, d in GROUPS if not d and g[0] == v])].max_conf.to_numpy()
            pa = T3[T3.group.isin([g for g, _, d in GROUPS if d and g[0] == v])].max_conf.to_numpy()
            r["auc"][v] = auc(p, n); r["auc"][v + "_any"] = auc(pa, n)
        th = r["t3"]["threshold"]; r["merged"] = {}                         # 원고 표 11 단위로 묶은 값(이미지 수로 가중)
        for k, gs, dr in MERGED:
            q = T3[T3.group.isin(gs)]; r["merged"][k] = float(((q.max_hit if dr else q.max_conf) >= th).mean())
    L += RP.table("**표 0. 원고 표 11: 취득 조건을 합친 재현율과 FPR (단위: %, fold 균형 평균)**", R, lambda r, k: r["merged"][k], [(k, n) for k, n in MERGED_NAMES])
    L += RP.table("**표 0-2. 표 0의 범위 (단위: %, 모델 중 가장 낮은 값과 가장 높은 값)**", R, lambda r, k: r["merged"][k], [(k, n) for k, n in MERGED_NAMES], rng=True)
    J["merged"] = {tc: {k: (lambda x: None if x is None else dict(mean=x[0], lo=x[1], hi=x[2], n=x[3]))(RP.agg(R, tc, lambda r, k=k: r["merged"][k])) for k, _, _ in MERGED} for tc, _ in RP.TRAIN}
    # IQ 파일 단위 bootstrap(2,000회, 원고 5.4절): 묶음마다 IQ 파일을 복원 추출 → 이미지 수 가중 비율 → 모델 fold 균형 평균 → 2.5·97.5 백분위수
    nimg = dict(zip(K["clip"], K.n_images)); rng = np.random.default_rng(0); J["bootstrap"] = {}
    L += ["**표 0-3. 표 0의 IQ 파일 단위 부트스트랩 95 % 신뢰구간 (단위: %, 2,000회)**", "", "취득 조건마다 IQ 파일을 복원 추출하고, 같은 표본을 모델 모두에 써서 fold 균형 평균을 다시 낸다.", "",
          "| 학습 조건 | " + " | ".join(n for _, n in MERGED_NAMES) + " |", "|---|" + "---:|" * len(MERGED)]
    for tc, tn in RP.TRAIN:
        q = [r for r in R if r["train"] == tc]
        if not q: continue
        folds = sorted({r["fold"] for r in q}); cells = []
        for k, gs, _ in MERGED:
            clips = sorted(c for g in gs for c in q[0]["t3"]["groups"][g]["clips"]); w = np.array([nimg[c] for c in clips], float)
            M = np.array([[r["t3"]["groups"][g]["clips"][c] for g in gs for c in sorted(r["t3"]["groups"][g]["clips"])] for r in q])   # 모델 × IQ 파일
            idx = rng.integers(0, len(clips), (2000, len(clips))); W = np.zeros((2000, len(clips)))
            np.add.at(W, (np.repeat(np.arange(2000), len(clips)), idx.ravel()), w[idx].ravel())
            P = (W @ M.T) / W.sum(1, keepdims=True)                                                                                   # 반복 × 모델
            fb = np.mean([P[:, [i for i, r in enumerate(q) if r["fold"] == f]].mean(1) for f in folds], axis=0)
            lo, hi = np.percentile(fb, [2.5, 97.5]); cells.append(f"{100 * lo:.1f}–{100 * hi:.1f}"); J["bootstrap"].setdefault(tc, {})[k] = [float(lo), float(hi)]
        L.append(f"| {tn} | " + " | ".join(cells) + " |")
    L.append("")
    VEN = [("S", "수원역 대합실"), ("T", "전철 1호선 객차")]; J["auc"] = {}
    for k, title, expl in (("", "**표 8-2. 장소별 AUC, 재현율·FPR 정의 그대로 (단위 없음, 0.5 = 구분 능력 없음, 1 = 완전한 구분, fold 균형 평균, 괄호는 범위)**",
                            "드론 존재 이미지의 점수 = Neo 2 채널에 맞는 드론 경계 상자의 최고 신뢰도, 드론 부재 이미지의 점수 = 위치와 상관없는 드론 경계 상자의 최고 신뢰도. 판정 임계값을 바꿔 가며 그린 재현율·FPR 곡선 아래 넓이와 같다."),
                           ("_any", "**표 8-3. 장소별 AUC, 양쪽 모두 위치와 상관없는 점수 (단위 없음, fold 균형 평균, 괄호는 범위)**",
                            "두 이미지 모두 위치와 상관없는 드론 경계 상자의 최고 신뢰도로 비교한다. 모델 출력이 드론 유무에 반응하는지만 본다.")):
        L += [title, "", expl, "", "| 학습 조건 | " + " | ".join(n for _, n in VEN) + " | 모델 수 |", "|---|---:|---:|---:|"]
        for tc, tn in RP.TRAIN:
            a = [RP.agg(R, tc, lambda r, v=v: r["auc"][v + k]) for v, _ in VEN]
            if all(x is None for x in a): continue
            J["auc"].setdefault(tc, {}).update({v + k: dict(mean=x[0], lo=x[1], hi=x[2]) for (v, _), x in zip(VEN, a)})
            L.append(f"| {tn} | " + " | ".join(f"{x[0]:.3f} ({x[1]:.3f}–{x[2]:.3f})" for x in a) + f" | {a[0][3]} |")
        L.append("")
    ga = lambda r, g: r["t3"]["groups"][g]["any_box"]
    L += RP.table("**표 9. 드론 존재 취득 조건에서 위치와 상관없이 드론 경계 상자가 하나라도 판정 임계값 이상인 이미지의 비율 (단위: %, 표 5와 비교용)**", R, ga, pos)
    L += ["## 3. 학습에서 뺀 계열별 결과", "", "fold 마다 시드 2개의 평균. 행 = 학습에서 뺀 계열(그 계열의 기종은 학습·검증 데이터에 없음).", ""]
    for tc, tn in RP.TRAIN:
        L += [f"**표 10-{tc}. {tn} 조건 모델: 학습에서 뺀 계열별 재현율과 FPR (단위: %)**", "",
              "| 학습에서 뺀 계열 | " + " | ".join(f"재현율: {n}" for _, n in pos) + " | " + " | ".join(f"FPR: {n}" for _, n in neg) + " |", "|---|" + "---:|" * (len(pos) + len(neg))]
        for fk, fn in RP.FOLDS:
            q = [r for r in R if r["train"] == tc and r["fold"] == fk]
            if q: L.append(f"| {fn} | " + " | ".join(f"{100 * np.mean([get(r, g) for r in q]):.1f}" for g, _, _ in GROUPS) + " |")
        L.append("")
    L += ["## 4. IQ 파일별 결과", "", "**표 11. IQ 파일별 재현율(드론 존재)과 FPR(드론 부재) (단위: %, fold 균형 평균, 열 = 학습 조건)**", "",
          "| 취득 조건 | IQ 파일 | " + " | ".join(tn for _, tn in RP.TRAIN) + " |", "|---|---|" + "---:|" * len(RP.TRAIN)]
    J["clips"] = {}
    for _, k in K.iterrows():
        a = [RP.agg(R, tc, lambda r, c=k["clip"], g=k["group"]: r["t3"]["groups"][g]["clips"].get(c)) for tc, _ in RP.TRAIN]
        J["clips"][k["clip"]] = {tc: (x[0] if x else None) for (tc, _), x in zip(RP.TRAIN, a)}
        L.append(f"| {GNAME[k.group]} | {short(k['clip'])} | " + " | ".join("" if x is None else f"{100 * x[0]:.1f}" for x in a) + " |")
    J["groups"] = {g: {tc: (lambda x: None if x is None else dict(mean=x[0], lo=x[1], hi=x[2], n=x[3]))(RP.agg(R, tc, lambda r, g=g: get(r, g))) for tc, _ in RP.TRAIN} for g, _, _ in GROUPS}
    J["any_box"] = {g: {tc: (lambda x: None if x is None else x[0])(RP.agg(R, tc, lambda r, g=g: ga(r, g))) for tc, _ in RP.TRAIN} for g, _ in pos}
    J["by_fold"] = {tc: {fk: {g: float(np.mean([get(r, g) for r in R if r["train"] == tc and r["fold"] == fk])) for g, _, _ in GROUPS}
                         for fk, _ in RP.FOLDS if any(r["train"] == tc and r["fold"] == fk for r in R)} for tc, _ in RP.TRAIN}
    (OUT / "RESULTS_test1_4.md").write_text("\n".join(L) + "\n", encoding="utf-8"); json.dump(J, open(OUT / "results_test1_4.json", "w"), ensure_ascii=False, indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "clips": clips()
    elif cmd == "eval": evaluate()
    elif cmd == "report": report()
    else: sys.exit(__doc__)
