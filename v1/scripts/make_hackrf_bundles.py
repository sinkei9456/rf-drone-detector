"""Build the HackRF recording bundles that are attached to the GitHub release.

Only the IQ files and the small metadata files that the code reads are kept:
  *.cs8 + *.cs8.json (indoor recordings, HackRF collection tool)
  iq.sigmf-data + iq.sigmf-meta (park recordings, SDRangel SigMF)
Any JSON key named "serial" (the HackRF device serial number) is replaced by "REDACTED".
Each bundle is an uncompressed tar (int8 IQ does not compress) smaller than 2 GB, to be extracted at the repository root:
  tar xf hackrf_1_train_background.tar   ->   data/raw/hackrf/...

Usage: python scripts/make_hackrf_bundles.py --src <folder that contains 1_nodrone_indoor, 2_neo2_indoor, 3_neo2_park> --out <output folder>
"""
import argparse, hashlib, json, shutil, tarfile
from pathlib import Path

BUNDLES = {
    "hackrf_1_train_background": ["1_nodrone_indoor/20261006_antenna_off", "3_neo2_park/02_telescopic_OFF_near1m"],
    "hackrf_2_neo2_park_train": ["3_neo2_park/01_telescopic_ON_near1m", "3_neo2_park/03_telescopic_distance10m_controller2m",
                                 "3_neo2_park/09_wifi_altitude10m_controller2m", "3_neo2_park/10_wifi_altitude10m_controller8m"],
    "hackrf_3_test_indoor": ["1_nodrone_indoor/20261007_antenna_on", "2_neo2_indoor/20261008_drone_off", "2_neo2_indoor/20261008_drone_on"],
    "hackrf_4_test_park_a": ["3_neo2_park/04_telescopic_distance50m_first_valid", "3_neo2_park/05_telescopic_distance50m_remeasure",
                             "3_neo2_park/06_telescopic_altitude50m_controller2m", "3_neo2_park/07_wifi_altitude30m_controller2m"],
    "hackrf_5_test_park_b": ["3_neo2_park/08_wifi_altitude50m_controller2m_retry", "3_neo2_park/11_wifi_horizontal50m_altitude40m",
                             "3_neo2_park/12_wifi_horizontal50m_altitude20m", "3_neo2_park/13_wifi_freeflight_unknown_geometry"],
}
KEEP = ("*.cs8", "*.cs8.json", "iq.sigmf-data", "iq.sigmf-meta")
LIMIT = 2 * 1000 ** 3            # GitHub release asset limit is 2 GiB; stay under 2 GB


def redact(o):
    if isinstance(o, dict): return {k: ("REDACTED" if k.lower() == "serial" else redact(v)) for k, v in o.items()}
    if isinstance(o, list): return [redact(v) for v in o]
    return o


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    src, out = Path(a.src), Path(a.out); out.mkdir(parents=True, exist_ok=True); sums = []
    for name, folders in BUNDLES.items():
        stage = out / "stage" / name; shutil.rmtree(stage, ignore_errors=True); n = 0
        for rel in folders:
            files = sorted({f for pat in KEEP for f in (src / rel).rglob(pat)})
            assert files, f"no recordings in {src / rel}"
            for f in files:
                dst = stage / "data/raw/hackrf" / f.relative_to(src); dst.parent.mkdir(parents=True, exist_ok=True)
                if f.name.endswith(".json") or f.name.endswith(".sigmf-meta"):
                    d = json.load(open(f, encoding="utf-8-sig")); json.dump(redact(d), open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                else:
                    shutil.copyfile(f, dst)
                n += 1
        tar = out / f"{name}.tar"
        with tarfile.open(tar, "w") as t: t.add(stage / "data", arcname="data")
        size = tar.stat().st_size; assert size < LIMIT, f"{tar} is {size / 1e9:.2f} GB"
        sums.append(f"{sha256(tar)}  {tar.name}"); shutil.rmtree(stage)
        print(f"{tar.name}: {n} files, {size / 1e9:.2f} GB")
    (out / "SHA256SUMS_hackrf.txt").write_text("\n".join(sums) + "\n"); shutil.rmtree(out / "stage", ignore_errors=True)
    print("BUNDLES_DONE")


if __name__ == "__main__":
    main()
