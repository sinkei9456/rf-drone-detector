"""DRFF-R2 2.4 GHz 녹음(dataset6-wifi_mixed 72개 + dataset7 indoor_environment) → 20 MHz × 20 ms 그림과 3종류 라벨 (2026-10-05).
DRFF-R2 는 파일 단위 정보(기종·기체 번호·비행 조건)만 있고 박스가 없다. 그래서 박스를 이렇게 만든다.
  drone : 파일마다 드론 영상 신호의 주파수 자리(band)를 1.4초 전체의 '켜져 있는 비율'로 자동으로 찾고, 사람이 확인 그림으로 검사
          (틀리면 runs/review/drff_band_fix.json 에 고침). 그림마다 band 안의 신호 덩어리 처음~끝 × band(창 안으로 자름) 를 drone 박스로.
          band 가 창 안에 4 MHz 미만만 걸치면 drone 박스를 그리지 않고 그 덩어리는 라벨 없이 둔다.
  wifi·bt: **라벨을 달지 않는다(방식 A, 10-05 사용자 결정).** 자동 규칙(넓으면 wifi, 좁으면 bt)을 Mendeley 사람 라벨에 대 보니
          wifi 는 사람 박스의 0.45 만 찾고 규칙 박스의 0.08 만 맞았고, bt 는 0.31 / 0.33 이었다(ruleacc, runs/review/drff_ruleacc.txt).
          DRFF 그림에서도 그림 대부분을 덮는 bt 박스, 드론 신호 가장자리를 wifi 로 단 박스가 보였다. 틀린 라벨을 넣느니 비워 둔다
          (A·B300 에서 DroneDetect·Tampere 그림을 drone 박스만으로 쓴 것과 같은 방식). label_from_bursts 의 wifi·bt 부분은 ruleacc 측정용으로만 남김.
  드론 신호 가장자리만 걸친 그림(band 가 창 안에 0 초과 4 MHz 미만, 실제로는 구간 #4 의 아래 끝 2 MHz): drone 박스를 그릴 수도, 비워 둘 수도 없어
          학습셋·검증셋·테스트셋 어디에도 넣지 않고 split_v4/edge_drff.txt 에 따로 적는다(10장 규칙 시험에서 따로 보고).
  덩어리 찾기는 자동 규칙(src/linklabel.detect_bursts) 그대로.
셋 나누기(기체 단위): Mini 4 Pro·Air 2S·Air 2 → 1·2호기 학습셋, 3호기 검증셋, 4호기 테스트셋 / Mini 3 Pro → 1호기 학습셋, 2호기 테스트셋 /
  1대뿐인 Mini 5 Pro·Air 3S(mavic3S)·Mavic 3·Mavic 3 Classic(mavic3C) → 조건 c1·c2 학습셋, c4 검증셋, c6 테스트셋 / 드론 없는 실내 배경 → 테스트셋.
그림: 파일마다 20 ms 창 10개(0.14초 간격) × 시연 구간 #1~#4(가운데 2.410·2.430·2.450·2.470 GHz). #5 는 녹음 범위(2.387~2.487 GHz) 밖.
단계:
  python src/drff_prep.py scan      # 파일마다: 켜짐 비율·band 자동 찾기, 그림(PNG)·덩어리 목록 저장, 확인 그림 (받은 파일만, 이미 한 것은 건너뜀)
  python src/drff_prep.py sheet     # band 확인 그림 모음 runs/review/drff_band_sheet_*.png
  python src/drff_prep.py label     # band(+고침) 로 라벨·목록(split_v4/{train,val,test}_drff.txt) 쓰기
  python src/drff_prep.py ruleacc   # 같은 wifi·bt 규칙을 Mendeley 사람 라벨에 적용해 정확도(사람 drone 박스 자리를 band 로 줌)
  python src/drff_prep.py check     # 라벨 겹쳐 그린 표본 80장 runs/review/drff_label_check_*.png
"""
import sys, json, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import linklabel
from spectrogram import spectrogram_db, to_image

RAW = ROOT / "data/raw/drff_r2"; OUT = ROOT / "data/derived/yolo_drff_3cls"; REV = ROOT / "runs/review"
FS_IN, FS, WIN = 100e6, 20e6, 0.02; N = int(FS * WIN); NFFT, HOP = 8192, int(FS * 0.1 / 490)
SLICES = {"s1": 2.410e9, "s2": 2.430e9, "s3": 2.450e9, "s4": 2.470e9}
T0S = [0.005 + 0.14 * k for k in range(10)]
MIN_IN_MHZ = 4.0


def files():
    fs = sorted((RAW / "dataset6-wifi_mixed").glob("*.mat")); env = RAW / "dataset7-environment/indoor_environment.mat"
    want = {l.split()[2].split("/")[-1]: int(l.split()[1]) for l in open(RAW / "dl_list.txt")}
    fs = [f for f in fs if f.stat().st_size == want.get(f.name, -1)]                 # 다 받은 파일만
    return fs + ([env] if env.exists() else [])


def split_of(stem):
    if stem.startswith("indoor_environment"): return "test"
    m, u, c = re.match(r"(\w+?)_(\d+)_(c\d+)", stem).groups(); u = int(u)
    if m in ("mini4PRO", "mavicAir2s", "mavicAir2"): return {1: "train", 2: "train", 3: "val", 4: "test"}[u]
    if m == "mini3pro": return {1: "train", 2: "test"}[u]
    return {"c1": "train", "c2": "train", "c4": "val", "c6": "test"}[c]


def load(p):
    import h5py
    with h5py.File(p, "r") as f:
        fc = float(f["CenterFrequence" if "CenterFrequence" in f else "CenterFreq"][()].ravel()[0]); fs = float(f["Fs"][()].ravel()[0])   # 실외 배경 파일은 CenterFreq
        x = np.asarray(f["RF0_I"][:, 0]) + 1j * np.asarray(f["RF0_Q"][:, 0])
    assert fs == FS_IN, fs
    return x.astype(np.complex64), fc


def occupancy(x, fc):
    """0.2 ms 마다 1024 표본(10 µs)을 떼어 스펙트럼 → 주파수 칸마다 '바닥 + 10 dB 넘는 비율'. 반환 (주파수 Hz 절대값, 비율, 100 ms 덩어리별 비율 [14×칸])."""
    step, nf = 20000, 1024; n = len(x) // step
    X = np.fft.fftshift(np.fft.fft(x[: n * step].reshape(n, step)[:, :nf] * np.hanning(nf), axis=1), axes=1)
    Pdb = 10 * np.log10(np.abs(X) ** 2 + 1e-30)
    floor = np.median(np.percentile(Pdb, 10, axis=0)); on = Pdb > floor + 10
    f = fc + np.fft.fftshift(np.fft.fftfreq(nf, 1 / FS_IN))
    chunks = on[: n // 500 * 500].reshape(-1, 500, nf).mean(1)
    return f, on.mean(0), chunks, Pdb


def find_band(f, occ, frac=0.8):
    """켜짐 비율(1 MHz 평활)의 가장 큰 연속 구간(최대값의 frac 배 이상, 폭 5 MHz 이상). 0.5 배로는 이웃 신호까지 붙어 넓게 잡혀 0.8 배로 바꿈(10-05). 반환 (f0, f1, 평균 비율) 또는 None, 그리고 다음 후보."""
    from scipy.ndimage import median_filter
    k = 10; o = np.convolve(median_filter(occ, 21), np.ones(k) / k, mode="same");          # 가운데 값 걸러 좁은 단일 톤(2.400 GHz 의 바늘)을 지움 o[(f < f.min() + 2e6) | (f > f.max() - 2e6)] = 0
    th = frac * o.max(); runs, i = [], 0
    while i < len(o):
        if o[i] >= th:
            j = i
            while j + 1 < len(o) and o[j + 1] >= th: j += 1
            if f[j] - f[i] >= 5e6: runs.append((float(o[i:j + 1].sum()), float(f[i]), float(f[j]), float(o[i:j + 1].mean())))
            i = j + 1
        else: i += 1
    runs.sort(reverse=True)
    return [(r[1], r[2], r[3]) for r in runs[:2]]


def scan_one(p):
    from scipy.signal import resample_poly
    p = Path(p); stem = p.stem; meta = REV / f"drff_scan/{stem}.json"
    if meta.exists(): return stem, "skip"
    x, fc = load(p)
    f, occ, chunks, Pdb = occupancy(x, fc)
    cand = find_band(f, occ)
    np.savez_compressed(REV / f"drff_scan/{stem}_occ.npz", f=f, occ=occ, chunks=chunks, spec=Pdb[:1000].astype(np.float16))
    (OUT / "images").mkdir(parents=True, exist_ok=True); rec = []
    for wi, t0 in enumerate(T0S):
        s0 = int(t0 * FS_IN) - 20000; seg = x[s0: s0 + int(WIN * FS_IN) + 40000]
        tt = (np.arange(len(seg)) + s0) / FS_IN
        for sn, c in SLICES.items():
            y = resample_poly((seg * np.exp(-2j * np.pi * (c - fc) * tt)).astype(np.complex64), 1, 5).astype(np.complex64)[4000: 4000 + N]
            S, _, _ = spectrogram_db(y, FS, nfft=NFFT, hop=HOP); P = 10 ** (S.astype(np.float64) / 10)
            name = f"drff_{stem}_{sn}_w{wi}"
            to_image(S).save(OUT / "images" / f"{name}.png")
            b, _ = linklabel.detect_bursts(P, FS / P.shape[0], WIN / P.shape[1], f_min=-FS / 2)
            rec.append(dict(name=name, slice=sn, center=c, window=wi, t0=t0, bursts=[list(map(float, q)) for q in b]))
    json.dump(dict(stem=stem, fc=fc, split=split_of(stem), cand=cand, images=rec), open(meta, "w"))
    return stem, f"cand {[(round(a / 1e9, 4), round(b / 1e9, 4), round(m, 2)) for a, b, m in cand]}"


def label_from_bursts(bursts, band_w, span):
    """bursts: (f0,f1,t0,t1,fill) 창 좌표(가운데 0 Hz). band_w: 창 안으로 자른 드론 자리 (f0,f1) 또는 None. 반환 links(cls 0 drone/1 wifi/2 bt), 무시한 덩어리 수."""
    links, ignored = [], 0
    def ov(b, band): return max(0.0, min(b[1], band[1]) - max(b[0], band[0])) / max(b[1] - b[0], 1e-9)
    drone, rest = [], []
    for b in bursts:
        if band_w is not None and ov(b, band_w) >= 0.5: drone.append(b)
        else: rest.append(b)
    if band_w is not None and drone:
        if band_w[1] - band_w[0] >= MIN_IN_MHZ * 1e6:
            links.append(dict(cls=0, f0=band_w[0], f1=band_w[1], t0=min(b[2] for b in drone), t1=max(b[3] for b in drone)))
        else: ignored += len(drone)
    wide = [b for b in rest if b[1] - b[0] >= linklabel.WIDE_MHZ * 1e6]; narrow = [b for b in rest if b[1] - b[0] < linklabel.WIDE_MHZ * 1e6]
    used = [False] * len(wide)
    for i, b in enumerate(wide):
        if used[i]: continue
        grp = [b]; used[i] = True
        for j in range(i + 1, len(wide)):
            if not used[j] and linklabel._overlap(b[:2], wide[j][:2]) >= 0.8: grp.append(wide[j]); used[j] = True
        links.append(dict(cls=1, f0=float(np.median([g[0] for g in grp])), f1=float(np.median([g[1] for g in grp])), t0=min(g[2] for g in grp), t1=max(g[3] for g in grp)))
    inside = lambda b: any(L["cls"] == 1 and b[0] >= L["f0"] - 0.1e6 and b[1] <= L["f1"] + 0.1e6 and b[2] >= L["t0"] and b[3] <= L["t1"] for L in links)
    narrow = [b for b in narrow if not inside(b)]
    if narrow: links.append(dict(cls=2, f0=min(b[0] for b in narrow), f1=max(b[1] for b in narrow), t0=min(b[2] for b in narrow), t1=max(b[3] for b in narrow)))
    return links, ignored


BAND_SEARCH = (2.435e9, 2.470e9)   # 10-05 사람 확인: 72개 확인 그림에서 보이는 드론 영상 신호는 모두 2.443~2.462 GHz 안. 전체에서 찾으면
                                   # 2.400 GHz 아래 늘 바쁜 띠(녹음 아래 끝)나 와이파이를 고르는 일이 있어(Mini 3 Pro 8개) 이 범위에서만 찾는다.
MIN_ON = 0.40                      # 고른 자리의 켜짐 비율이 이보다 낮으면 영상 신호가 없는 것(신호 있는 파일은 0.47 이상, Mini 4 Pro 3호기 0.31~0.32)


def band_info():
    """파일마다 dict(band=(f0,f1) 또는 None, on=켜짐 비율, how=auto/fix/excluded/env). 고침은 runs/review/drff_band_fix.json(사람 확인, 이유 '_why')."""
    fix = json.load(open(REV / "drff_band_fix.json")) if (REV / "drff_band_fix.json").exists() else {}
    out = {}
    for m in sorted((REV / "drff_scan").glob("*.json")):
        d = json.load(open(m)); st = d["stem"]
        if st.startswith("indoor_environment"): out[st] = dict(band=None, on=None, how="env"); continue
        z = np.load(REV / f"drff_scan/{st}_occ.npz"); f = z["f"]
        cand = find_band(f, np.where((f >= BAND_SEARCH[0]) & (f <= BAND_SEARCH[1]), z["occ"], 0.0))   # 저장한 켜짐 비율로 다시 찾음
        auto = tuple(cand[0][:2]) if cand else None; on = cand[0][2] if cand else 0.0
        if st in fix: out[st] = dict(band=tuple(fix[st]) if fix[st] else None, on=on, how="fix" if fix[st] else "excluded")
        else: out[st] = dict(band=auto, on=on, how="auto")
    return out


def bands():
    return {k: v["band"] for k, v in band_info().items()}


def excluded():
    """영상 신호가 없어 어느 셋에도 넣지 않는 드론 녹음(사람 확인)."""
    return sorted(k for k, v in band_info().items() if v["how"] == "excluded")


def cmd_label():
    B = bands(); EX = set(excluded()); (OUT / "labels").mkdir(parents=True, exist_ok=True); lists = {"train": [], "val": [], "test": [], "edge": []}; stat = {}
    for m in sorted((REV / "drff_scan").glob("*.json")):
        d = json.load(open(m)); st = d["stem"]; band = B[st]
        if st in EX: stat.setdefault("excluded_files", []).append(st); continue                  # 영상 신호 없음 → 어느 셋에도 안 넣음
        for r in d["images"]:
            bw = None
            if band is not None:
                lo, hi = band[0] - r["center"], band[1] - r["center"]; lo, hi = max(lo, -FS / 2), min(hi, FS / 2)
                bw = (lo, hi) if hi > lo else None
            links, ign = label_from_bursts([tuple(b) for b in r["bursts"]], bw, FS)
            links = [L for L in links if L["cls"] == 0]                                     # 방식 A: drone 박스만
            lines = linklabel.links_to_yolo(links, FS, WIN, f_min=-FS / 2)
            with open(OUT / "labels" / f"{r['name']}.txt", "w") as fh:
                for c, xc, yc, w, h in lines: fh.write(f"{c} {xc:.5f} {yc:.5f} {w:.5f} {h:.5f}\n")
            edge = bw is not None and (bw[1] - bw[0]) < MIN_IN_MHZ * 1e6                       # 드론 신호 가장자리만 걸침
            key = "edge" if edge else d["split"]
            lists.setdefault(key, []).append(str(OUT / "images" / f"{r['name']}.png"))
            s = stat.setdefault(key, dict(img=0, img_with_drone=0, drone_boxes=0, files=set())); s["img"] += 1; s["files"].add(st)
            s["drone_boxes"] += len(lines); s["img_with_drone"] += len(lines) > 0
    for k, v in lists.items(): open(ROOT / f"data/derived/split_v4/{k}_drff.txt", "w").write("\n".join(sorted(v)) + "\n")
    for k, v in stat.items():
        if isinstance(v, dict): v["files"] = len(v["files"])
    print("그림·박스 수:", json.dumps(stat, ensure_ascii=False)); json.dump(stat, open(REV / "drff_label_stat.json", "w"))


def cmd_sheet():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    B = bands(); stems = sorted(B); per = 12
    for k in range(0, len(stems), per):
        sub = stems[k:k + per]; fig, ax = plt.subplots(len(sub), 2, figsize=(16, 2.6 * len(sub)), squeeze=False, gridspec_kw=dict(width_ratios=[1, 2.2]))
        for r, st in enumerate(sub):
            z = np.load(REV / f"drff_scan/{st}_occ.npz"); f = z["f"] / 1e9; band = B[st]
            d = json.load(open(REV / f"drff_scan/{st}.json"))
            ax[r][0].plot(f, z["occ"], lw=0.8); ax[r][0].set_ylim(0, 1); ax[r][0].set_title(f"{st} [{d['split']}] on-fraction per freq", fontsize=8)
            ax[r][1].imshow(z["spec"].astype(np.float32).T, aspect="auto", origin="lower", cmap="gray", extent=[0, 200, f[0], f[-1]],
                            vmin=np.percentile(z["spec"], 5), vmax=np.percentile(z["spec"], 99.5))
            ax[r][1].set_title("first 200 ms (sparse spectrogram), red = drone band used for labels", fontsize=8)
            for s, c in SLICES.items(): ax[r][1].axhline(c / 1e9 - 0.01, color="#3b82f6", lw=0.5, ls=":")
            if band:
                for a in (ax[r][0],): a.axvspan(band[0] / 1e9, band[1] / 1e9, color="r", alpha=0.15)
                for v in band: ax[r][1].axhline(v / 1e9, color="r", lw=1)
            ax[r][0].tick_params(labelsize=7); ax[r][1].tick_params(labelsize=7)
        fig.tight_layout(); fig.savefig(REV / f"drff_band_sheet_{k // per + 1}.png", dpi=55); plt.close(fig)
        print("sheet", k // per + 1, sub[0], "...", sub[-1])


def cmd_ruleacc():
    """Mendeley 테스트 그림(사람 라벨)에서: 그림 → 세기 → 같은 덩어리 찾기 → 사람 drone 박스 자리를 band 로 주고 wifi·bt 박스 → 사람 wifi·bt 와 IoU ≥ 0.3 비교."""
    from PIL import Image
    span_f, span_t = 56e6, 0.02
    imgs = [l.strip() for l in open(ROOT / "data/derived/split_v4/test_mendeley.txt") if l.strip()]; imgs = imgs[:: max(1, len(imgs) // 600)][:600]
    def iou(p, q):
        ax0, ay0, ax1, ay1 = p[0] - p[2] / 2, p[1] - p[3] / 2, p[0] + p[2] / 2, p[1] + p[3] / 2
        bx0, by0, bx1, by1 = q[0] - q[2] / 2, q[1] - q[3] / 2, q[0] + q[2] / 2, q[1] + q[3] / 2
        inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0)); return inter / (p[2] * p[3] + q[2] * q[3] - inter + 1e-9)
    res = {1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}; img_ok = {1: 0, 2: 0}; n = 0
    for p in imgs:
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0; P = (10 ** (g * 50 / 10))[::-1]
        b, _ = linklabel.detect_bursts(P, span_f / P.shape[0], span_t / P.shape[1], f_min=-span_f / 2)
        lab = [l.split() for l in open(p.replace("/images/", "/labels/").replace(".png", ".txt")) if l.strip()]
        H = [(int(l[0]), *map(float, l[1:5])) for l in lab]
        dr = [h for h in H if h[0] == 0]
        band = None
        if dr:   # 사람 drone 박스의 주파수 범위(합집합 근사: 가장 넓게)
            lo = min(span_f * (0.5 - (h[2] + h[4] / 2)) for h in dr); hi = max(span_f * (0.5 - (h[2] - h[4] / 2)) for h in dr); band = (lo, hi)
        links, _ = label_from_bursts(b, band, span_f)
        R = [(c, *box) for c, *box in linklabel.links_to_yolo(links, span_f, span_t, f_min=-span_f / 2)]
        for c in (1, 2):
            h = [x[1:] for x in H if x[0] == c]; r = [x[1:] for x in R if x[0] == c]
            res[c][0] += sum(any(iou(a, q) >= 0.3 for q in r) for a in h); res[c][1] += len(h)
            res[c][2] += sum(any(iou(a, q) >= 0.3 for q in h) for a in r); res[c][3] += len(r)
            img_ok[c] += (len(h) > 0) == (len(r) > 0)
        n += 1
    out = [f"Mendeley 테스트 {n}장, 사람 drone 박스 자리를 알려 준 채 wifi·bt 규칙 적용 (IoU ≥ 0.3)"]
    for c, nm in ((1, "wifi"), (2, "bt")):
        a, A, b2, B2 = res[c]
        out.append(f"  {nm}: 사람 박스 중 규칙이 찾음 {a}/{A} = {a / max(A, 1):.2f} | 규칙 박스 중 사람 박스와 겹침 {b2}/{B2} = {b2 / max(B2, 1):.2f} | 그림 단위 있음/없음 일치 {img_ok[c] / n:.2f}")
    print("\n".join(out)); open(REV / "drff_ruleacc.txt", "w").write("\n".join(out) + "\n")


def cmd_check():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    rng = np.random.default_rng(0); pick = []
    for sp, k in (("train", 32), ("val", 16), ("test", 32)):
        L = [l.strip() for l in open(ROOT / f"data/derived/split_v4/{sp}_drff.txt") if l.strip()]
        pick += [(sp, p) for p in rng.choice(L, size=min(k, len(L)), replace=False)]
    col = {0: "#ef4444", 1: "#22c55e", 2: "#3b82f6"}
    for s in range(0, len(pick), 20):
        fig, ax = plt.subplots(4, 5, figsize=(20, 17))
        for a, (sp, p) in zip(ax.ravel(), pick[s:s + 20]):
            a.imshow(np.asarray(Image.open(p)), cmap="gray", aspect="auto")
            for l in open(p.replace("/images/", "/labels/").replace(".png", ".txt")):
                c, xc, yc, w, h = l.split(); c = int(c); xc, yc, w, h = (float(v) * 640 for v in (xc, yc, w, h))
                a.add_patch(Rectangle((xc - w / 2, yc - h / 2), w, h, fill=False, ec=col[c], lw=1.6))
            a.set_title(f"[{sp}] {Path(p).stem.replace('drff_', '')}", fontsize=8); a.set_xticks([]); a.set_yticks([])
        for a in ax.ravel()[len(pick[s:s + 20]):]: a.set_axis_off()
        fig.suptitle("DRFF-R2 auto labels: red = drone, green = wifi, blue = bt (20 MHz vertical, high freq on top; 20 ms horizontal)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(REV / f"drff_label_check_{s // 20 + 1}.png", dpi=50); plt.close(fig)
    print("check sheets", (len(pick) + 19) // 20)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan":
        (REV / "drff_scan").mkdir(parents=True, exist_ok=True); fs = files()
        with ProcessPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 3) as ex:
            for st, msg in ex.map(scan_one, map(str, fs)): print(st, msg, flush=True)
    else: dict(sheet=cmd_sheet, label=cmd_label, ruleacc=cmd_ruleacc, check=cmd_check)[cmd]()
