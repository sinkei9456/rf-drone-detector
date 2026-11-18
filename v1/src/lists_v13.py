"""새 분석(10-07 오후): 학습·검증은 합성 v8(잘린 신호 포함)만. 테스트는 실제 데이터를 20 MHz 로 가공한 것 + 합성 v8 테스트.
  (다) da: 학습 = 합성 v8 학습 6,000 / 검증 = 합성 v8 검증 600
  테스트(평가에만): 합성 v8 테스트 1,000(split_v6/syn8_test.txt), DRFF-R2 20 MHz 보이는 비율별(split_v6/drff8_*.txt, src/drff_v8_test.py)
출력: data/derived/split_v6/{da_train,da_val,syn8_test}.txt, data/yaml/v13_da.yaml, v13_test_syn.yaml, v13_test_drff.yaml"""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data/derived/split_v6"; D.mkdir(exist_ok=True); SYN = ROOT / "data/derived/yolo_syn_iq_v8"
rd = lambda p: [l.strip() for l in open(p) if l.strip()]
st, sv, se = sorted(rd(SYN / "train.txt")), sorted(rd(SYN / "val.txt")), sorted(rd(SYN / "test.txt"))
for k, v in {"da_train": st, "da_val": sv, "syn8_test": se}.items(): (D / f"{k}.txt").write_text("\n".join(v) + "\n"); print(f"{k}: {len(v)}")
Y = "path: " + str(ROOT / "data/derived") + "\ntrain: split_v6/{tr}.txt\nval: split_v6/{va}.txt\nnames:\n  0: drone\n  1: wifi\n  2: bt\n"
(ROOT / "data/yaml/v13_da.yaml").write_text(Y.format(tr="da_train", va="da_val"))
(ROOT / "data/yaml/v13_test_syn.yaml").write_text(Y.format(tr="da_train", va="syn8_test"))        # 평가(val)에만 씀
(ROOT / "data/yaml/v13_test_drff.yaml").write_text(Y.format(tr="da_train", va="drff8_main"))      # 평가(val)에만 씀
print("LISTS_DONE")
