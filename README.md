# RF Drone Detector

English | [한국어](README.ko.md)

**Passive RF drone detection: a YOLO detector that finds drone video links in 20 ms spectrograms from a HackRF One, trained mostly on IQ-level synthetic pictures**

D4D x EDTH hackathon 2026, problem T1-3 "Small-drone and balloon intrusion over the capital area (Gray-zone C-UAS)". A small software-defined radio (HackRF One) listens to 20 MHz of the 2.4 GHz band. Every 20 ms of IQ samples becomes one 640 × 640 spectrogram, and a YOLO26s detector draws boxes around drone video links, Wi-Fi and Bluetooth. A drone box gives the time (ms) and frequency (MHz) of the drone signal. The detector only listens, so it does not interfere with anything around it.

This repository contains the code that produced the released model `model_02a.pt`, from the raw recordings to the trained weights and the test results, and the CPU deployment code.

![How to read a picture](docs/figures/how_to_read.png)

*One picture = 20 ms of a HackRF recording (DJI Neo 2 flying at 10 m in a park). Horizontal: time, 0–20 ms. Vertical: frequency, 20 MHz (top = higher). Brighter = stronger. Left: the picture the model sees (brightness stretched per picture). Right: the same 20 ms with a fixed brightness scale. The dashed yellow lines (drawn by us) mark the drone channel; the repeating bright rectangles between them are the drone video signal.*

## Method

**Why spectrograms.** Raw IQ (400,000 complex samples per 20 ms) does not show signals to the eye or to an image model. As a time × frequency × power picture, each signal type has its own shape: a drone video link is a 10–20 MHz wide rectangle repeating at a fixed interval, Wi-Fi is a short 20 MHz bar, Bluetooth a small 1–2 MHz patch. An off-the-shelf object detector can then find each signal and its time and frequency range, even when several signals overlap. Recordings with different sample rates (HackRF 20 MS/s, public datasets 100 MS/s) are resampled to the same 20 MHz × 20 ms grid. What is lost: phase (no RF fingerprinting of individual drones) and absolute power (each picture is normalized, so very weak signals look like noise texture).

**IQ-level synthesis.** Flying real drones at many heights and places and drawing every box by hand is slow and error-prone. Instead, pieces of real drone signals are cut from recordings and added to real HackRF background noise at the IQ level, then the picture is made once, so overlapping signals and noise look like a real receiver. The answer boxes come from where the pieces were placed.

![Synthesis steps](docs/figures/synthesis_steps.png)

*Building one synthetic picture: (1) background, real HackRF noise recorded in the park with drone and controller off; (2) DJI Neo 2 piece cut from a 10 m park recording; (3) the same piece weakened to a 30 m level (1.3 dB above noise) and switched on for only 2 ms each time; (4) 10 generated Wi-Fi bursts; (5) 25 generated Bluetooth bursts; (6) a piece of another drone (DJI, public DRFF-R2 data) moved up so that only part of it is inside the picture; (7) answer boxes made automatically (red drone, green Wi-Fi, blue Bluetooth).*

Situations put into the training pictures on purpose:

- Crowded spectrum: up to 12 Wi-Fi and 30 Bluetooth bursts per picture.
- Weak signals of drones flying higher or farther: Neo 2 pieces from 10 m recordings weakened to the 30 m level (1–2 dB above noise, measured in the park), with the on-time of each burst shortened to 1.5–3.5 ms for half of the flying pieces. The lower limit is 1 dB; weaker pieces would have boxes on signals that are not in the picture.
- Drone signals cut by the 20 MHz receiver window: at least 50 % of the signal width visible.
- Real "no drone" pictures (real Wi-Fi rooms and receiver noise) learned as "nothing to find".

**Detector.** Ultralytics YOLO26s, starting from the COCO object-detection weights (`yolo26s.pt`) and fine-tuning all layers on the spectrograms (transfer learning; 696 of 708 weight tensors are reused, only the class-dependent output parts are new). 40 epochs, batch 16, seed 0, image size 640, last-epoch weights. No comparison with training from scratch was made, so the benefit of COCO pre-training is not measured.

| Training data of model 2a | Train | Validation |
|---|---:|---:|
| Synthetic pictures (synthesis v10) | 6,000 | 600 |
| Real "no drone" pictures | 1,720 | 230 |
| Total | 7,720 | 830 |

- Drone pieces: 303 from public data (DRFF-R2, Tampere, DroneDetect, RFUAV; 158 train, 51 validation, 94 test-only) and 1,643 DJI Neo 2 pieces from our park recordings (1,205 train, 438 validation).
- Backgrounds: half indoor HackRF noise (antenna removed, Bluetooth bursts erased), half park recording with drone off.
- Synthetic training pictures: 3,558 with a drone (552 with two), 1,512 with only Wi-Fi/Bluetooth, 930 background only.

## Results

All test pictures come from recordings that were not used for training or validation. A drone counts as found when the model draws a drone box with confidence ≥ 0.25 that overlaps the answer box enough (IoU ≥ 0.3). Released model `model_02a.pt`:

| Test | Found / pictures | Rate |
|---|---:|---:|
| HackRF, DJI Neo 2 on the ground indoors (2–3 m) | 249 / 250 | 99.6 % |
| HackRF, Neo 2 free flight in the park | 496 / 497 | 99.8 % |
| HackRF, Neo 2 at 30 m altitude (signal visible in the picture) | 350 / 351 | 99.7 % |
| HackRF, Neo 2 at about 50 m (signal visible in the picture) | 178 / 183 | 97.3 % |
| DRFF-R2, 8 recent DJI models (other receiver), drone signal 100 / 80 / 60 / 50 % visible | | 99.5 / 99.0 / 98.0 / 86.4 % |
| Synthetic test, pictures with a drone (same synthesis as training, test-only pieces) | 508 / 519 | 97.9 % |

| False-alarm test (pictures without a drone) | Drone box drawn / pictures |
|---|---:|
| HackRF receiver noise (antenna off and on) | 0 / 350 |
| DRFF-R2 rooms with real Wi-Fi | 6 / 440 |
| HackRF indoor, Neo 2 switched off | 0 / 250 |
| Total | 6 / 1,040 (0.6 %) |

![Results of model 2a](docs/figures/results_model_2a.png)

*(a) Pictures with a drone found with the correct position (%): orange = HackRF Neo 2 tests, blue = DRFF-R2 DJI drones by visible fraction; numbers under the bars are picture counts. (b) Pictures without a drone that got a drone box (%). (c) Neo 2 pictures by signal strength above noise.*

Limits of these numbers:

- The park tests were recorded on the same day and at the same place as the Neo 2 training pieces; the indoor test (other place and time) is the fairest.
- The 30 m number should be trusted less: the on-time shortening was chosen after looking at the shape of that 30 m test recording.
- At about 50 m the Neo 2 signal is only 0.4 dB above noise; 2,989 park pictures where the signal is not visible cannot be scored.
- Only 2.4 GHz. Drones that do not transmit (or balloons without radios) cannot be found.
- Wi-Fi and Bluetooth boxes of model 2a are not reliable (real Wi-Fi was also learned as "nothing to find"); use the drone boxes only.

## Reproduction check

The whole pipeline was re-run with `scripts/reproduce.sh` on 2026-10-11 in a fresh copy of this repository, on one NVIDIA GB10 in the original analysis environment (not a newly built one; its full package list is `requirements-lock.txt`), with the HackRF recordings taken from the release files and only the public-data subsets listed under "Data". Every intermediate file was compared byte by byte with the files used for the released model.

| Step | Result |
|---|---|
| HackRF backgrounds, drone pieces (303 from public data, 1,643 Neo 2), Neo 2 pictures and automatic labels | identical |
| Synthetic pictures v8 (7,600) and test 2 pictures (1,840), with labels | identical |
| Real "no drone" pictures (1,950) and synthetic training pictures v10 (6,600), with labels | identical |
| Training of model 2a: losses and validation values of all 40 epochs | identical |
| Trained weights (708 tensors) | identical (largest difference 0.0); the `.pt` file differs only in metadata such as date and folder |
| Tests 1–4 of the reproduced model and of the released model | identical to the numbers in "Results" (test 4 also picture by picture) |

- Time: about 1.5 hours, of which training is 54 minutes. Disk: 155 GB of raw data, 13 GB of files made by the pipeline.
- One input is not recomputed: the automatic DRFF-R2 band candidates. The current `drff_prep.py scan` gives slightly different candidates for 14 of 73 files than when the bands were reviewed by a person, so `scripts/reproduce.sh` uses the reviewed candidates in `data/labels/drff_scan_reviewed/` (the per-file signal arrays computed by the scan were identical).
- Identical numbers need the same software versions and GPU type; on other hardware small numerical differences are expected.
- At the start of training Ultralytics downloads `yolo26n.pt` for its mixed-precision check, so the training machine needs internet access.

## Repository layout

```
rf-drone-detector/
├── data/
│   ├── labels/                      human-reviewed DRFF-R2 drone bands (part of the labels)
│   │   ├── drff_band_fix.json       bands corrected by a person
│   │   └── drff_scan_reviewed/      automatic band candidates that were reviewed (73 files)
│   └── lists/drff_r2_files.txt      DRFF-R2 files used (fileId, size, path)
├── deploy/                          CPU deployment (Raspberry Pi 64-bit OS or any Linux PC)
│   ├── models/model_02a.pt          released model
│   ├── samples/                     40 pictures to check an installation
│   ├── samples_expected/model_02a/  results of model_02a.pt on samples/ on our server
│   ├── src/iq_to_image.py           recording (.cs8 or SigMF) -> 20 ms pictures
│   ├── src/predict.py               pictures -> drone boxes, CSV, summary
│   └── environment.yml              conda environment (CPU torch)
├── docs/figures/                    figures used in this README
├── scripts/
│   ├── download_drff_r2.sh          download the 73 DRFF-R2 files
│   ├── make_hackrf_bundles.py       how the HackRF release bundles were made
│   └── reproduce.sh                 raw recordings -> model_02a -> tests 1-4
├── src/                             pipeline code (comments are in Korean)
├── LICENSE                          AGPL-3.0
├── README.md / README.ko.md
├── requirements.txt                 training environment (requirements-lock.txt = full pip freeze)
└── requirements-lock.txt
```

`src/` is research code kept as it was run; several files contain earlier versions (for example `synth_iq_build.py --ver v2 … v11`). Model 2a uses pieces `v3` → `v7`, test synthesis `v8` and training synthesis `v10`.

## Use the released model (CPU)

```bash
cd deploy
conda env create -f environment.yml          # Python 3.11, CPU torch 2.14.0, ultralytics 8.4.157
conda activate d4d_detector
# check the installation: the drone yes/no column must match our server
python src/predict.py --images samples --model models/model_02a.pt --out output/check --device cpu
diff <(cut -d, -f1,8 output/check/pictures.csv) <(cut -d, -f1,8 samples_expected/model_02a/pictures.csv) && echo "same as server"
# your own HackRF recording (.cs8 with its .json, or a SigMF folder), 2.4 GHz, 20 MS/s
python src/iq_to_image.py --input <recording file or folder> --out work/images
python src/predict.py --images work/images --model models/model_02a.pt --device cpu
```

On a Raspberry Pi use a 64-bit OS (torch has no 32-bit ARM wheel). On our ARM server CPU one picture took about 60–75 ms to make and 0.35–0.4 s to run through the model.

## Reproduce the model

1. **Environment.** The analysis environment (training, synthesis, evaluation) was:

   | Item | Version |
   |---|---|
   | OS, CPU | Ubuntu 24.04.4 LTS, aarch64 (ARM 64-bit) |
   | GPU | NVIDIA GB10, driver 580.126.09, CUDA 13.0 |
   | Environment manager | conda 25.9.1, environment with Python 3.11.16 |
   | Main packages | torch 2.14.0 (CUDA 13.0 wheel), torchvision 0.29.0, ultralytics 8.4.157, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, pillow 12.3.0, h5py 3.16.0, matplotlib 3.11.2 |
   | Full list | `requirements-lock.txt` (pip freeze of that environment) |

   ```bash
   conda create -n rfdd python=3.11.16 -y && conda activate rfdd
   pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130   # other GPUs: the torch 2.14.0 wheel for your CUDA
   pip install -r requirements.txt
   ```

   The reproduction check above was run in the original analysis environment; an environment built from these files has the same versions but was not separately re-run. Put the COCO-pretrained `yolo26s.pt` from the release into `src/` (the exact file used; SHA-256 `646f8bc3…`).
2. **Data** into `data/raw/` (about 150 GB of public data + 5.6 GB of our recordings, see below).
3. **Run** `bash scripts/reproduce.sh` from the repository root. Steps write logs to `runs/logs/`; finished steps are skipped when it is run again. Training takes about 1 hour on one NVIDIA GB10.

## Data

No data is stored in the git repository.

| Folder | Dataset | What is used | Size | Source and license |
|---|---|---|---:|---|
| `data/raw/drff_r2/` | DRFF-R2 (8 recent DJI models, USRP, 100 MS/s, 2.437 GHz) | `dataset6-wifi_mixed/` 72 files + `dataset7-environment/indoor_environment.mat` | 48 GB | Science Data Bank, CC BY 4.0. `bash scripts/download_drff_r2.sh` |
| `data/raw/tampere/` | Tampere University drone RF recordings | the 12 files `*_2G*.bin` | 5.4 GB | Zenodo, doi:10.5281/zenodo.4264467 (see the record for its license) |
| `data/raw/dronedetect/DroneDetect_V2/CLEAN/` | DroneDetect V2 (Swinney and Woods) | the `CLEAN` folder | 85 GB | IEEE DataPort (see the dataset page for its terms) |
| `data/raw/rfuav/` | RFUAV | folders `DJI MINI3/` and `DJI MINI4 PRO/` | 12.5 GB | RFUAV benchmark dataset (see its page for its terms) |
| `data/raw/hackrf/` | Our HackRF One recordings (2.45 GHz, 20 MS/s) | all | 5.6 GB | GitHub release, 5 tar files, extract at the repository root. CC BY 4.0 |

HackRF release bundles (extract with `tar xf <file>` at the repository root; checksums in `SHA256SUMS_hackrf.txt`):

| File | Contents | Used for |
|---|---|---|
| `hackrf_1_train_background.tar` | indoor, antenna removed (5 × 1 s); park, drone off | synthetic backgrounds, real "no drone" pictures, test 1 |
| `hackrf_2_neo2_park_train.tar` | park sessions 01 (1 m), 03, 09, 10 (10 m) | Neo 2 training and validation pieces |
| `hackrf_3_test_indoor.tar` | indoor, antenna on, no drone; indoor Neo 2 on / off | tests 1 and 4 |
| `hackrf_4_test_park_a.tar` | park sessions 04, 05, 06 (about 50 m), 07 (30 m) | test 4 |
| `hackrf_5_test_park_b.tar` | park sessions 08, 11, 12 (about 50 m), 13 (free flight) | test 4 |

Only the IQ files and the metadata read by the code are included; the HackRF device serial number is replaced by `REDACTED` (`scripts/make_hackrf_bundles.py`).

## License

Code and model: AGPL-3.0 (`LICENSE`), because the code uses Ultralytics YOLO (AGPL-3.0) and the model is fine-tuned from Ultralytics weights.
Our HackRF recordings (release files `hackrf_*.tar`): CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); please cite this repository.
The public datasets keep their own licenses and are not redistributed here.
