"""rev1 원고(2026-11-12)의 본문·부록 그림. paper/src/fig_rev2.py 가 이 파일의 함수를 불러 쓴다. 그림 안 글자는 한국어(나눔고딕).
본문 그림(paper/fig/rev1_f*.png)
  design    : 그림 1. 실험 설계 (rev1a 카드형 레이아웃 참고. 학습 조건 5개 = 간섭 밀도 4조건 + 혼합(1:1:1:1, 11-13 추가), 모델 50개)
  concept   : 그림 2. 도심 배치 개념과 탐지 장치의 처리 흐름 (rev1a 참고)
  picture   : 그림 3. 입력 이미지(공원 세션 09, 파일 1, 2.48 s). 왼쪽 = 모델 입력, 오른쪽 = 고정 밝기
  synthesis : 그림 4. 합성 이미지 한 장을 만드는 과정 (공원 무신호 + 드론 신호 세그먼트 + 버스트)
  density   : 그림 5. 같은 이미지를 간섭 밀도 4조건(대조군, 20 %, 50 %, 80 %)으로 만든 예. 버스트는 포함 관계(밀도가 높을수록 앞 조건의 버스트를 모두 포함)
부록 그림(paper/fig/rev1_<부록 문자><번호>_*.png)
  models    : 그림 B-1. 공개 20개 기종의 드론 신호 세그먼트(fold 별 한 줄, 공원 무신호에 SNR 15 dB 로 얹음)
  filesplit : 그림 B-2. IQ 파일 단위 group split 예(DJI Phantom 4)
  neo2      : 그림 C-1. 실측 Neo 2 IQ 데이터의 신호 모양(세션 01·09·07)
  noise     : 그림 D-1. 배경 노이즈 두 종류(수신기 노이즈 15 MHz 필터, 공원 무신호 20 MHz 필터)와 평균 스펙트럼
  baseline  : 그림 E-1. 1-3(옛 baseline) 실측 IQ 데이터와 합성 이미지 비교
사용: python paper/src/fig_rev1.py [이름 ...]   (이름 없이 실행하면 전부)"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, FancyArrowPatch
ROOT = Path(__file__).resolve().parents[2]; FIG = ROOT / "paper/fig"
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "paper/src"))
import synth_iq as S
import piece_tables as P
import hackrf_paths as HP


def nanum():
    """시스템에 설치된 나눔고딕(NanumGothic.ttf, NanumGothicBold.ttf)을 matplotlib 에 등록하고 글꼴 이름을 돌려준다. 없으면 기본 글꼴(한글이 깨짐)."""
    fs = {Path(f).name: f for f in fm.findSystemFonts()}
    for n in ("NanumGothic.ttf", "NanumGothicBold.ttf"):
        if n in fs: fm.fontManager.addfont(fs[n])
    if "NanumGothic.ttf" in fs: return fm.FontProperties(fname=fs["NanumGothic.ttf"]).get_name()
    print("나눔고딕이 없어 기본 글꼴로 그린다(한글이 깨질 수 있음). Ubuntu: sudo apt install fonts-nanum", file=sys.stderr); return plt.rcParams["font.family"]


plt.rcParams.update({"font.family": nanum(), "axes.unicode_minus": False, "font.size": 10,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "svg.fonttype": "path", "pdf.fonttype": 42})
# 개념도 색 (rev1a 와 같은 계열)
INKD, MUTED, LINE, PALE = "#17273a", "#576b7b", "#d5e0e6", "#f3f7f9"
BLUE, AMBER, TEAL, PURPLE, RED = "#2468a2", "#cb7436", "#167e79", "#7459a4", "#bf4b4b"
BOXC = {"0": "red", "1": "lime", "2": "deepskyblue"}
FOLDS = [("Mini", ["DJI Mini 5 Pro", "DJI Mini 4 Pro", "DJI Mini 3 Pro", "DJI Mini 3", "DJI Mavic Mini"]),
         ("Air", ["DJI Air 3S", "DJI Air 2S", "DJI Mavic Air 2", "DJI Mavic Air"]),
         ("Mavic", ["DJI Mavic 3", "DJI Mavic 3 Classic", "DJI Mavic Pro", "DJI Mavic Pro 2"]),
         ("전문가용", ["DJI Phantom 4", "DJI Inspire 2", "DJI Matrice 100", "DJI Matrice 210"]),
         ("타 제조사", ["Parrot Disco", "Parrot Mambo", "Yuneec Typhoon H"])]
HC = ROOT / "data/derived/hackrf_clean"
BG_ORIG = S.make_background_v8          # 원래 버스트 생성 함수(바꿔 끼우기 전의 것을 따로 잡아 둠)
T_LAB, F_LAB = "시간 (ms)", "주파수 오프셋 (MHz)"
PARK = "Park02_1"                        # 공원 무신호 train 구간(0–1 s). 그림 예시의 배경 노이즈


def save(fig, name, **k):
    fig.savefig(FIG / name, dpi=k.pop("dpi", 140), facecolor="white", **k); plt.close(fig); print(name)


def show(a, img, xl=True, yl=True):
    a.imshow(img, cmap="gray", vmin=0, vmax=255, extent=(0, 20, -10, 10), aspect="auto"); a.set_xticks([0, 10, 20]); a.set_yticks([-10, -5, 0, 5, 10])
    if xl: a.set_xlabel(T_LAB)
    if yl: a.set_ylabel(F_LAB)


def draw_boxes(a, bx, lw_d=1.6, lw_o=0.9):
    """bx = [(종류, t0 s, t1 s, f_lo Hz, f_hi Hz)]"""
    for k, t0, t1, lo, hi in bx:
        c = {"drone": "red", "wifi": "lime", "bt": "deepskyblue"}[k]
        a.add_patch(Rectangle((t0 * 1e3, lo / 1e6), (t1 - t0) * 1e3, (hi - lo) / 1e6, fill=False, ec=c, lw=lw_d if k == "drone" else lw_o))


def bg_iq(name, off=5_000_000):
    """배경 노이즈 20 ms (노이즈 전력 1 로 맞춤)"""
    a = np.load(HC / f"{name}.npy", mmap_mode="r"); off = min(off, len(a) - S.N); x = np.asarray(a[off: off + S.N]).astype(np.complex64); x = x - x.mean()
    return (x / np.sqrt(np.mean(np.abs(x) ** 2))).astype(np.complex64)


def pieces():
    return np.load(ROOT / "runs/iq_pieces_v7.npy", mmap_mode="r"), json.load(open(ROOT / "runs/iq_pieces_v7_meta.json"))


def on_noise(arr, meta, j, snr=15.0):
    p = dict(meta[j]); p["iq"] = np.array(arr[j]); x, _, _ = S.add_drone(bg_iq(PARK), p, np.random.default_rng(0), snr, fo=0.0)
    return S.to_gray_v7(x)


def park_imgs(session):
    import occupancy as E
    out = []
    for f in sorted((HP.NEO2_PARK / session / "captures").glob("*/iq.sigmf-data")): out += E.read_iq(f)
    return out


def occupancy(x, floor_from, drone_boxes):
    """드론 박스를 뺀 스펙트럼 점유율. floor_from = 노이즈 플로어를 잴 IQ(버스트를 넣기 전)."""
    import occupancy as E
    yolo = [((t0 + t1) / 2 / S.WIN, 1 - ((lo + hi) / 2 + S.FS / 2) / S.FS, (t1 - t0) / S.WIN, (hi - lo) / S.FS) for _, t0, t1, lo, hi in drone_boxes]
    fl = np.percentile(E.power(floor_from), 10, axis=0); return E.occ_parts(E.power(x), fl, yolo)[2]


# ---------------- 개념도 공통 ----------------
def card(ax, x, y, w, h, title, body, color=BLUE, face="white", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", facecolor=face, edgecolor=LINE, linewidth=1))
    ax.plot([x + .02, x + .02], [y + .025, y + h - .025], color=color, lw=3, solid_capstyle="round")
    ax.text(x + .038, y + h - .035, title, fontsize=fs + 1.5, weight="bold", color=INKD, va="top")
    ax.text(x + .038, y + h - .085, body, fontsize=fs, color=MUTED, va="top", linespacing=1.6)


def arrow(ax, p, q, c=MUTED):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color=c))


# ---------------- 그림 1 ----------------
def design():
    """11-16: 1번 묶음(조용한 배경, 모델 50개, 평가 1-1–1-4)과 2번 묶음(혼잡한 실제 배경, 평가 2-1·2-2)을 위아래 두 줄로 그린다."""
    fig, ax = plt.subplots(figsize=(15, 10.4)); ax.set(xlim=(0, 1), ylim=(0, 1.42)); ax.axis("off"); o = .38
    for x, t in [(.02, "01  데이터"), (.355, "02  합성 데이터와 학습"), (.69, "03  평가")]: ax.text(x, 1.418, t, fontsize=13, weight="bold", color=BLUE, va="top")
    ax.text(.02, 1.362, "1번 묶음  |  조용한 배경(공원 무신호)에 합성 드론과 합성 간섭을 넣어 학습", fontsize=12, weight="bold", color=INKD, va="top")
    card(ax, .02, .61 + o, .29, .30, "공개 드론 IQ 데이터", "20개 기종  ·  드론 신호 세그먼트 303개\n제품 계열별 5개 fold\n4개 fold 학습  /  1개 fold 테스트\n테스트 fold 기종은 학습에 쓰지 않음\n테스트 fold를 바꿔 5회 반복", BLUE, PALE)
    card(ax, .02, .33 + o, .29, .22, "공원 무신호", "합성 이미지의 배경 노이즈\n0–1 s train  /  1–2 s validation\n2–3 s test", TEAL, PALE)
    card(ax, .02, .05 + o, .29, .22, "DJI Neo 2", "학습·검증에 쓰지 않음\n평가 1-2·1-3·1-4, 2-1·2-2에만 사용", PURPLE, "#f8f6fb")
    card(ax, .355, .61 + o, .29, .30, "합성 데이터", "드론 신호 세그먼트 + 공원 무신호\n+ 생성한 와이파이·블루투스 버스트\ntrain 6,000장  /  validation 600장\nIQ 파일 단위 group split", TEAL, PALE)
    card(ax, .355, .33 + o, .29, .22, "학습 조건 5개 (간섭 밀도)", "대조군 2 %  /  약 20 %  /  중 50 %  /  강 80 %\n혼합: 네 조건 1:1:1:1\n스펙트럼 점유율 중앙값 기준", BLUE)
    card(ax, .355, .05 + o, .29, .22, "YOLO26s 학습", "5조건 × 5 fold × 2 seed\n= 모델 50개\nconfidence threshold: validation에서 튜닝", BLUE, "#f1f6fb")
    card(ax, .69, .74 + o, .29, .17, "1-1  |  미학습 계열 (합성)", "테스트 fold의 드론 신호 세그먼트\n+ 합성 간섭, 간섭 밀도 4단계", AMBER, "#fffaf6", 9.5)
    card(ax, .69, .51 + o, .29, .19, "1-2  |  Neo 2 (합성)", "Neo 2 드론 신호 세그먼트\n+ 합성 간섭, 간섭 밀도 4단계", PURPLE, "#faf8fc", 9.5)
    card(ax, .69, .28 + o, .29, .19, "1-3  |  간섭이 적은 실제 환경", "공원·실내 실측 IQ 데이터\nNeo 2 켬과 끔", PURPLE, "#faf8fc", 9.5)
    card(ax, .69, .05 + o, .29, .19, "1-4  |  실제 혼잡 장소", "수원역 대합실: 지상 영상 켜짐, 비행\n전철 1호선 객차: 지상 영상 켜짐\n드론을 끈 IQ 데이터로 FPR", PURPLE, "#faf8fc", 9.5)
    arrow(ax, (.31, .77 + o), (.35, .77 + o))
    ax.plot([.31, .333, .333], [.44 + o, .44 + o, .66 + o], color=MUTED, lw=1.4); arrow(ax, (.333, .66 + o), (.35, .66 + o))
    arrow(ax, (.5, .61 + o), (.5, .555 + o)); arrow(ax, (.5, .33 + o), (.5, .275 + o))
    ax.plot([.645, .665], [.16 + o, .16 + o], color=MUTED, lw=1.2); ax.plot([.665, .665], [.145 + o, .825 + o], color=MUTED, lw=1.2)
    for y in (.825, .605, .375, .145): arrow(ax, (.665, y + o), (.686, y + o))
    ax.plot([.02, .98], [.405, .405], color=LINE, lw=1.2, ls="--")
    ax.text(.02, .385, "2번 묶음  |  혼잡한 장소의 실제 배경에 합성 드론을 넣어 학습", fontsize=12, weight="bold", color=INKD, va="top")
    card(ax, .02, .02, .29, .30, "혼잡한 장소의 드론 끔 IQ 데이터", "1-4와 같은 날 취득, 장소별로 나눠 씀\n객차: 세션 6개 train, 2개 validation\n수원역: 영상 켜기 전 train,\n영상 끈 뒤 validation", TEAL, PALE)
    card(ax, .355, .02, .29, .30, "실제 배경 + 합성 드론", "공개 20개 기종의 드론 신호 세그먼트\n+ 실제 혼잡 배경 (간섭을 더하지 않음)\n드론 한 클래스, 2 seed, threshold는\n학습 장소의 validation에서 튜닝\n대조: 공원 배경 + 합성 간섭", BLUE, "#f1f6fb")
    card(ax, .69, .19, .29, .13, "2-1  |  객차 배경으로 학습", "수원역 데이터로 평가", PURPLE, "#faf8fc", 9.5)
    card(ax, .69, .02, .29, .13, "2-2  |  수원역 배경으로 학습", "객차 데이터로 평가", PURPLE, "#faf8fc", 9.5)
    arrow(ax, (.31, .17), (.35, .17)); ax.plot([.645, .665], [.17, .17], color=MUTED, lw=1.2); ax.plot([.665, .665], [.085, .255], color=MUTED, lw=1.2)
    for y in (.255, .085): arrow(ax, (.665, y), (.686, y))
    ax.text(.69, -.025, "보라색 = DJI Neo 2 데이터(평가에만 사용)", fontsize=9.5, color=PURPLE)
    save(fig, "rev1_f1_design.png", dpi=200, bbox_inches="tight", pad_inches=.12)


# ---------------- 그림 2 ----------------
def concept():
    fig = plt.figure(figsize=(15, 6.4)); ax = fig.add_axes([.025, .12, .55, .76]); ax.set(xlim=(0, 100), ylim=(0, 64), aspect="equal"); ax.axis("off")
    ax.text(0, 66, "(a) 도심 배치 개념", fontsize=14, weight="bold", color=INKD)
    ax.add_patch(Rectangle((0, 0), 100, 64, color="#edf2f4"))
    ax.add_patch(Rectangle((0, 24), 100, 16, color="white", ec="none"))
    for x in (20, 48, 76): ax.add_patch(Rectangle((x, 0), 8, 64, color="white", ec="none"))
    for x in (2, 30, 58, 86):
        for y in (3, 43):
            ax.add_patch(FancyBboxPatch((x, y), 12, 17, boxstyle="round,pad=0,rounding_size=1", fc="#cbd8de", ec="none"))
            for dx, dy in ((3.5, 5), (8.5, 12)): ax.plot(x + dx, y + dy, "^", color=TEAL, ms=7, mec="white", mew=.7)
    for x in (12, 37, 62, 87):
        ax.add_patch(Circle((x, 32), 13, fc=BLUE, alpha=.06, ec="none")); ax.add_patch(Circle((x, 32), 13, fill=False, ec=BLUE, ls=(0, (3, 3)), lw=1))
        ax.plot(x, 32, "s", color=INKD, ms=8, mec="white", mew=1)
    rng = np.random.default_rng(5)
    for x, y in zip(rng.uniform(1, 99, 26), rng.uniform(25.5, 38.5, 26)): ax.plot(x, y, "o", color=AMBER, ms=3.5)
    ax.plot(66, 44, marker="X", color=RED, ms=12, mec="white", mew=1); arrow(ax, (65.4, 42), (63, 34.5), RED)
    hd = [plt.Line2D([], [], marker="s", ls="none", color=INKD, label="탐지 장치"), plt.Line2D([], [], ls="--", color=BLUE, label="탐지 범위(개념)"),
          plt.Line2D([], [], marker="^", ls="none", color=TEAL, label="와이파이 공유기"), plt.Line2D([], [], marker="o", ls="none", color=AMBER, label="휴대폰·이어폰 등 블루투스 기기"),
          plt.Line2D([], [], marker="X", ls="none", color=RED, label="드론")]
    ax.legend(handles=hd, loc="upper left", bbox_to_anchor=(0, -.03), ncol=5, frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.0, borderaxespad=0)
    ax.text(0, -10, "장치 배치와 탐지 범위는 개념을 나타낸 것이며, 축척이나 실측 성능이 아니다.", fontsize=9, color=MUTED)
    b = fig.add_axes([.61, .12, .37, .76]); b.set(xlim=(0, 1), ylim=(0, 1)); b.axis("off"); b.text(0, 1.04, "(b) 탐지 장치 한 대의 처리 흐름", fontsize=14, weight="bold", color=INKD)
    ent = [("01", "수신", "안테나 + HackRF One  ·  20 MHz 대역"), ("02", "IQ 구간 분할", "20 ms  ·  복소 샘플 400,000개"), ("03", "이미지 변환", "스펙트로그램  ·  640 × 640"),
           ("04", "객체 검출", "YOLO26s  ·  드론 / 와이파이 / 블루투스"), ("05", "판정", "드론 박스의 confidence ≥ threshold (validation에서 튜닝)")]
    for i, (n, t, s) in enumerate(ent):
        y = .82 - i * .19
        b.add_patch(FancyBboxPatch((0, y), .98, .14, boxstyle="round,pad=0.01,rounding_size=0.02", fc=PALE, ec=LINE, lw=1))
        b.text(.035, y + .07, n, fontsize=11, weight="bold", color=TEAL, va="center"); b.text(.13, y + .095, t, fontsize=11.5, weight="bold", color=INKD, va="center")
        b.text(.13, y + .042, s, fontsize=9.5, color=MUTED, va="center")
        if i < 4: arrow(b, (.5, y - .005), (.5, y - .045))
    save(fig, "rev1_f2_concept.png", dpi=200, bbox_inches="tight", pad_inches=.12)


# ---------------- 그림 3 ----------------
def picture():
    f = sorted((HP.NEO2_PARK / "09_wifi_altitude10m_controller2m/captures").glob("*/iq.sigmf-data"))[0]; k0 = int(2.48 * S.FS) * 2
    r = np.fromfile(f, dtype=np.int8, count=2 * S.N, offset=k0).astype(np.float32); x = (r[0::2] + 1j * r[1::2]).astype(np.complex64)
    Sd = S.spectrogram_v7(x); nf = float(np.median(Sd)); fixed = np.clip((Sd - nf + 6) / 24, 0, 1)[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6)); ext = (0, 20, 2439.5, 2459.5)
    imgs = [(S.to_gray_v7(x), 0, 255, "(a) 모델 입력: 이미지마다 밝기를 정규화"), (fixed, 0, 1, "(b) 같은 20 ms를 고정 밝기로 표시\n(노이즈 = 짙은 회색, 노이즈보다 18 dB 이상 = 흰색)")]
    for a, (im, v0, v1, t) in zip(ax, imgs):
        a.imshow(im, cmap="gray", vmin=v0, vmax=v1, extent=ext, aspect="auto"); a.set_xlabel(T_LAB); a.set_ylabel("주파수 (MHz)"); a.set_title(t, fontsize=10.5)
        a.set_yticks([2440, 2444.5, 2449.5, 2454.5, 2459]); a.set_xticks([0, 5, 10, 15, 20])
        for y in (2444.5, 2454.5): a.axhline(y, color="#eda100", ls="--", lw=1.4)
        a.text(19.6, 2455.0, "노란 점선 = 드론 채널(저자가 덧그림)", ha="right", va="bottom", fontsize=8.5, color="#8a5d00", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        a.annotate("드론 영상 버스트\n(약 4.95 ms마다 반복)", xy=(5.8, 2450.5), xytext=(1.5, 2457.3), fontsize=9, color="#c0392b", weight="bold",
                   arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3), bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    fig.tight_layout(); save(fig, "rev1_f3_input.png")


# ---------------- 그림 4 ----------------
def synthesis():
    arr, meta = pieces(); bg = bg_iq(PARK, off=3_000_000); rng = np.random.default_rng(11)
    j = [k for k, m in enumerate(meta) if m["group"] == "Tampere DJI_inspire_2_2G 0.00-0.56s"][3]; p = dict(meta[j]); p["iq"] = np.array(arr[j])
    fo = -7.0e6; p["iq"] = S.cut_to_window(p["iq"], fo); xd, bd, _ = S.add_drone(np.zeros_like(bg), p, rng, 15.0, fo=fo)
    xw, bw, _ = BG_ORIG(np.random.default_rng(21), 8, 0, base=np.zeros_like(bg)); xb, bb, _ = BG_ORIG(np.random.default_rng(31), 0, 25, base=np.zeros_like(bg))
    steps = [("(1) 배경 노이즈\n(공원 무신호)", bg, []), ("(2) + 드론 신호 세그먼트(DJI Inspire 2)\n창의 아래 경계에 걸치게 배치", bg + xd, []),
             ("(3) + 생성한 와이파이 버스트 8개", bg + xd + xw, []), ("(4) + 생성한 블루투스 버스트 25개", bg + xd + xw + xb, []),
             ("(5) 자동으로 만든 정답 박스\n빨강 = 드론, 초록 = 와이파이, 파랑 = 블루투스", bg + xd + xw + xb, [bd] + bw + bb)]
    fig, ax = plt.subplots(1, 5, figsize=(18, 4.6))
    for i, (a, (t, x, bx)) in enumerate(zip(ax, steps)):
        show(a, S.to_gray_v7(x.astype(np.complex64)), yl=i == 0); a.set_title(t, fontsize=10); draw_boxes(a, bx)
    fig.tight_layout(); save(fig, "rev1_f4_synthesis.png")


# ---------------- 그림 5 ----------------
def density():
    arr, meta = pieces(); bg = bg_iq(PARK, off=7_000_000); z = np.zeros_like(bg)
    j = next(k for k, m in enumerate(meta) if m["group"] == "DRFF mavic3_1_c1"); p = dict(meta[j]); p["iq"] = np.array(arr[j])
    fo = 0.0; xd, bd, _ = S.add_drone(z, p, np.random.default_rng(3), 12.0, fo=fo)
    W = [BG_ORIG(np.random.default_rng([41, i]), 1, 0, base=z)[:2] for i in range(80)]          # 버스트 하나씩(밀도가 높을수록 앞 버스트를 모두 포함)
    B = [BG_ORIG(np.random.default_rng([43, i]), 0, 1, base=z)[:2] for i in range(240)]
    W = [w for w in W if w[1]]; B = [b for b in B if b[1]]
    def build(k):
        x = bg + xd + sum((w[0] for w in W[:k]), z) + sum((b[0] for b in B[:3 * k]), z)
        return x.astype(np.complex64), [bd] + [w[1][0] for w in W[:k]] + [b[1][0] for b in B[:3 * k]]
    targets = [("대조군(저밀도)", None), ("약", 0.20), ("중", 0.50), ("강", 0.80)]; out = []; k = 1
    for name, tg in targets:
        if tg is None:
            x, bx = build(k); out.append((name, tg, k, occupancy(x, bg + xd, [bd]), x, bx)); continue
        while True:
            x, bx = build(k); o = occupancy(x, bg + xd, [bd])
            if o >= tg - 0.01 or k >= len(W) or 3 * k >= len(B): break
            k += 1
        out.append((name, tg, k, o, x, bx))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.9))
    for i, (a, (name, tg, kk, o, x, bx)) in enumerate(zip(ax, out)):
        show(a, S.to_gray_v7(x), yl=i == 0); draw_boxes(a, [bd])
        a.set_title((f"{name}: 목표 {round(tg * 100)} %" if tg else name) + f"\n와이파이 {kk}개, 블루투스 {3 * kk}개, 스펙트럼 점유율 {100 * o:.1f} %", fontsize=10)
    fig.tight_layout(); save(fig, "rev1_f5_density.png")
    print([(n, kk, round(100 * o, 1)) for n, tg, kk, o, x, bx in out])


# ---------------- 부록 B ----------------
def models():
    arr, meta = pieces(); kinds = [P.kind(m["group"])[0] for m in meta]; fig, ax = plt.subplots(5, 5, figsize=(14, 15.6))
    for i, (fold, L) in enumerate(FOLDS):
        for j in range(5):
            a = ax[i, j]
            if j >= len(L): a.axis("off"); continue
            idx = [k for k, kd in enumerate(kinds) if kd == L[j]]; k = idx[len(idx) // 2]
            show(a, on_noise(arr, meta, k), xl=i == 4 or (j >= len(FOLDS[i + 1][1]) if i < 4 else True), yl=j == 0)
            a.set_title(f"{L[j]}\n드론 신호 세그먼트 {len(idx)}개\n이 예의 대역폭 {meta[k]['bw'] / 1e6:.1f} MHz", fontsize=9.5)
        ax[i, 0].text(-0.38, 0.5, f"Fold {i + 1}: {fold}", transform=ax[i, 0].transAxes, rotation=90, ha="center", va="center", fontsize=11.5, weight="bold")
    fig.tight_layout(rect=(0.03, 0, 1, 1)); save(fig, "rev1_B1_models.png", dpi=110)


def filesplit():
    arr, meta = pieces(); fig, ax = plt.subplots(2, 4, figsize=(13, 7.2))
    tam = [j for j, m in enumerate(meta) if m["group"].startswith("Tampere DJI_phantom_4_2G")]
    tam.sort(key=lambda j: float(meta[j]["src"].split()[-1][:-1])); pick = [tam[0], tam[len(tam) // 3], tam[2 * len(tam) // 3], tam[-1]]
    for a, j in zip(ax[0], pick):
        show(a, on_noise(arr, meta, j), xl=False, yl=a is ax[0, 0]); a.set_title(f"{meta[j]['src'].split()[-1][:-1]} s 지점", fontsize=10)
    dd = [("PHA_0000_01", "train"), ("PHA_0000_02", "train"), ("PHA_0000_03", "validation"), ("PHA_0000_04", "train")]
    for a, (stem, sp) in zip(ax[1], dd):
        j = next(k for k, m in enumerate(meta) if stem in m["group"]); show(a, on_noise(arr, meta, j), yl=a is ax[1, 0])
        a.set_title(f"IQ 파일 {stem}: {sp}", fontsize=10, color=AMBER if sp == "validation" else BLUE)
    fig.text(0.01, 0.985, "(a) IQ 파일 하나(Tampere, DJI Phantom 4)에서 자른 드론 신호 세그먼트 4개: 서로 거의 같으므로 파일 전체를 train에 넣는다", fontsize=11, va="top")
    fig.text(0.01, 0.50, "(b) 같은 기종의 다른 IQ 파일 4개(DroneDetect)에서 하나씩 가져온 드론 신호 세그먼트: 파일 단위로 train 또는 validation에 넣는다", fontsize=11, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.5); save(fig, "rev1_B2_filesplit.png")


# ---------------- 부록 C ----------------
def band_db(x, half=4.3e6):
    X = np.fft.fft(x - x.mean()); fr = np.fft.fftfreq(len(x), 1 / S.FS); X[(np.abs(fr) > half) | (np.abs(fr) < 0.1e6)] = 0
    e = np.convolve(np.abs(np.fft.ifft(X)) ** 2, np.ones(1024) / 1024, "same"); return 10 * np.log10(e / np.quantile(e, 0.1))


def neo2():
    cols = [("01_telescopic_ON_near1m", "비행하지 않음, 조종기와 약 1 m"), ("09_wifi_altitude10m_controller2m", "비행, 고도 10 m"), ("07_wifi_altitude30m_controller2m", "비행, 고도 30 m")]
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.4), gridspec_kw=dict(height_ratios=[1.5, 1])); t = np.arange(S.N) / S.FS * 1e3
    for j, (s, name) in enumerate(cols):
        L = park_imgs(s); x = L[len(L) // 2].astype(np.complex64); a = ax[0, j]; show(a, S.to_gray_v7(x), yl=j == 0)
        for y in (-5, 5): a.axhline(y, color="yellow", ls="--", lw=1)
        a.set_title(f"{name}\n(공원 세션 {s[:2]})", fontsize=10)
        b = ax[1, j]; b.plot(t, band_db(x), color=BLUE, lw=0.8); b.set_xlim(0, 20); b.set_ylim(-3, 30); b.set_xlabel(T_LAB); b.grid(color="#e4e3df", lw=0.6)
        if j == 0: b.set_ylabel("드론 채널 전력\n(노이즈 대비 dB)")
    fig.tight_layout(); save(fig, "rev1_C1_neo2.png")


# ---------------- 부록 D ----------------
def noise():
    fig = plt.figure(figsize=(14.5, 6.8)); gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.12, 1.5])
    rows = [("수신기 노이즈(안테나 분리, 수신 필터 15 MHz): 합성에 쓰지 않음", [("AntennaOff_01", "파일 01"), ("AntennaOff_03", "파일 03"), ("AntennaOff_04", "파일 04")]),
            ("공원 무신호(드론·조종기 OFF, 수신 필터 20 MHz): 합성에 사용", [("Park02_1", "0–1 s 구간: train"), ("Park02_2", "1–2 s 구간: validation"), ("Park02_3", "2–3 s 구간: test")])]
    for i, (rn, L) in enumerate(rows):
        for j, (f, t) in enumerate(L):
            a = fig.add_subplot(gs[i, j]); show(a, S.to_gray_v7(bg_iq(f)), xl=i == 1, yl=j == 0); a.set_title(f"{rn}\n{t}" if j == 0 else f"\n{t}", fontsize=10, loc="left")
    a = fig.add_subplot(gs[:, 4]); f = np.fft.fftshift(np.fft.fftfreq(4096, 1 / S.FS)) / 1e6
    for k, (name, c) in enumerate([(f"AntennaOff_0{n}", BLUE) for n in range(1, 6)] + [(f"Park02_{n}", AMBER) for n in range(1, 4)]):
        v = np.asarray(np.load(HC / f"{name}.npy", mmap_mode="r")[:4096 * 2000]).reshape(-1, 4096); v = v - v.mean()
        p = np.fft.fftshift(np.mean(np.abs(np.fft.fft(v * np.hanning(4096), axis=1)) ** 2, 0)); p = 10 * np.log10(p / np.median(p)); p[np.abs(f) < 0.1] = np.nan
        a.plot(f, p, color=c, lw=1, alpha=0.8, label={0: "수신기 노이즈, 파일 01–05", 5: "공원 무신호, 구간 1–3"}.get(k))
    a.set_xlabel(F_LAB); a.set_ylabel("평균 전력 (중앙값 대비 dB)"); a.set_title("배경 노이즈 파일별 평균 스펙트럼", fontsize=10, loc="left")
    a.grid(color="#e4e3df", lw=0.6); a.legend(frameon=False, fontsize=9.5, loc="upper center"); a.set_xlim(-10, 10)
    fig.tight_layout(); save(fig, "rev1_D1_noise.png")


# ---------------- 부록 E ----------------
def baseline():
    import occupancy as E
    band = [(0.5, 0.5, 1.0, 0.5)]
    src = [("공원, 드론 OFF (세션 02)", park_imgs("02_telescopic_OFF_near1m"), False),
           ("공원, Neo 2 비행 고도 10 m (세션 09)", park_imgs("09_wifi_altitude10m_controller2m"), True),
           ("실내, Neo 2 OFF", sum((E.read_iq(f) for f in sorted((HP.NEO2_INDOOR["OFF"] / "recordings").glob("*/*.cs8"))), []), False),
           ("실내, Neo 2 ON, 2–3 m, 비행하지 않음", sum((E.read_iq(f) for f in sorted((HP.NEO2_INDOOR["ON"] / "recordings").glob("*/*.cs8"))), []), True)]
    fig, ax = plt.subplots(1, 4, figsize=(14, 4.6)); out = []
    for j, (a, (name, L, drone)) in enumerate(zip(ax, src)):
        Ps = [E.power(x) for x in L]; floor = np.percentile(np.concatenate(Ps, 0), 10, axis=0)
        occ = [E.occ_parts(Pw, floor, band)[2] for Pw in Ps]; i = int(np.argsort(occ)[len(occ) // 2])
        show(a, S.to_gray_v7(L[i]), yl=j == 0)
        if drone:
            for y in (-5, 5): a.axhline(y, color="yellow", ls="--", lw=1)
        a.set_title(f"{name}\n드론 채널 밖 스펙트럼 점유율 {100 * np.median(occ):.1f} %\n(이미지 {len(L)}장의 중앙값)", fontsize=9.5); out.append((name, round(100 * float(np.median(occ)), 2)))
    fig.tight_layout(); save(fig, "rev1_E1_test1_3.png"); print(out)


ALL = ["design", "concept", "picture", "synthesis", "density", "models", "filesplit", "neo2", "noise", "baseline"]
if __name__ == "__main__":
    for k in (sys.argv[1:] or ALL): globals()[k]()
