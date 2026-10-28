# Focus Coach

A laptop webcam watches you while you study and notices when you pick up your phone.
At the end of a session you get a report: how many pickups, how much time lost, and a timeline.
Everything runs on your own computer. Video is never saved unless you ask for it.

Computer Vision mini project (Karnavati University, Sem VII, Course 71407001002 / 73407001002).

> **Screenshot:** [SCREENSHOT: add a picture of the live window and of a session report here]

Pipeline: webcam (OpenCV) → brightness check → CLAHE (only if dark) → YOLO26n finds `head` + `phone`
→ geometric rules decide the usage state → time smoothing → events → session report.

## Results

Usage classification ("is this person using a phone?") on held-out images, and detection and speed headlines.
This block is written by `python report/build_report.py` from the files in `results/`; do not edit it by hand.

<!-- RESULTS:START (written by report/build_report.py) -->

| Method (held-out images) | Accuracy | F1 |
|---|---|---|
| A-coco: any phone found, COCO model | 0.679 | 0.632 |
| A-ft: any phone found, fine-tuned model | 0.876 | 0.854 |
| A2-authors: FPI-Det authors' logic on our boxes | 0.862 | 0.843 |
| B-rules: our geometric rules | 0.873 | 0.852 |
| C-mobilenet: MobileNetV2 on the whole image | 0.736 | 0.706 |

Phone detection on the test split: mAP50 0.313 (COCO) to 0.869 (fine-tuned). Processing speed of the app pipeline: 38 FPS on 13th Gen Intel Core i7-13700HX (without the camera). Full tables: `results/eval/summary.md` and `report/report.md`.

<!-- RESULTS:END -->

The full report is `report/report.md`, viva preparation is `report/viva.md`, and every table is in `results/eval/summary.md`.

## Quickstart (Windows, PowerShell)

You need Python 3.11, Git, and about 10 GB of free disk space.

```powershell
git clone <this repository> focus-coach
cd focus-coach
git clone --depth 1 https://github.com/KvCgRv/FPI-Det third_party/FPI-Det
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Data (manual download).** Get the FPI-Det images with the Google Drive link in `third_party/FPI-Det/README.md` and extract
them into `data/raw/` so that you end up with `data/raw/reorganized_dataset/{train,val,test}`. Then build the dataset:

```powershell
python ml/prepare_dataset.py
```

**Model (not in the repository).** The trained weights are not committed. Either train them on Google Colab with
`ml/train_colab.ipynb` (see [docs/COLAB.md](docs/COLAB.md); about 1.5 hours on a free T4 GPU) and save the result as
`models/best.pt`, or copy an existing `best.pt` there. The COCO weights `models/yolo26n.pt` are downloaded by Ultralytics on first use.

**Reproduce the results** (about 30 minutes on a laptop CPU):

```powershell
python ml/splits.py
python ml/detections.py
python ml/tune_rules.py
python ml/evaluate.py
python ml/make_lowlight.py
python ml/lowlight_experiment.py
python -m app.benchmark
python -m app.benchmark --imgsz 480
python ml/evaluate.py
python report/make_sample_session.py
python report/make_figures.py
python report/build_report.py
```

The MobileNet baseline needs TensorFlow in its own environment. Run it between the two `evaluate.py` runs:

```powershell
py -3.11 -m venv .venv-tf
.\.venv-tf\Scripts\python.exe -m pip install -r requirements-tf.txt
.\.venv-tf\Scripts\python.exe ml\mobilenet_baseline.py
```

Tests: `pytest ml/tests`.

## Run the app

```powershell
python -m app.focus_coach                       # webcam 0
python -m app.focus_coach --source demo.mp4     # a recorded clip instead of the webcam
python -m app.benchmark                         # speed of the pipeline on this computer
```

| Key | Action |
|---|---|
| **S** | start, pause, resume the session |
| **E** | end the session and show the report |
| **D** | show or hide the detection boxes |
| **C** | cycle low-light enhancement: auto, on, off |
| **Q** | quit (asks whether to save if a session is running) |

Options: `--imgsz 480` for more speed, `--save-video out.mp4` to record the annotated picture (off by default),
`--autostart`, `--no-display` (scripted runs). A state counts only after it holds for about 1.5 s, so a quick glance at the
phone is ignored; no head for 3 s shows AWAY. Ending a session writes `events.csv`, `summary.json` and `report.png` to
`sessions/<date_time>/`. If the camera will not open, close other programs that use it (Teams, Zoom), check Windows Settings >
Privacy > Camera, or try `--source 1`. A guide for recording a backup demo clip is in [app/make_demo_clip.md](app/make_demo_clip.md).

## Run it in Colab (optional browser demo)

Anyone can try Focus Coach in a browser with no install: open the notebook in Colab, click **Run all**, and open the link it prints
(live webcam, video upload and single-photo tabs). Steps are in [docs/COLAB_DEMO.md](docs/COLAB_DEMO.md).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/focus-coach/blob/main/demo/focus_coach_colab.ipynb)
[replace YOUR-USERNAME with your GitHub name after you push the repo]

Locally: `pip install -r requirements-demo.txt`, then `python -m demo.gradio_app`. To process a clip without any window:
`python -m app.offline clip.mp4 --out sessions/offline_clip/`.

## Repository layout

```
CLAUDE.md          rules for working on the project         docs/        phase plans and PROGRESS.md
shared/rules.json  all thresholds and app settings           ml/          dataset, rules, smoothing, enhancement,
app/               the OpenCV desktop app                                 demo/        Gradio demo for Colab (optional)                                 tuning, evaluation, experiments, tests
results/           metrics, plots, failure grids            report/      report, viva notes, figures
models/ data/ sessions/ third_party/   not committed (downloads and your own data)
```

## Privacy

Frames are processed in memory and discarded. The app writes only event times and summary numbers to `sessions/`.
Saving the annotated video is opt-in (`--save-video`). Use it for yourself only, with your own consent.

## Credits and licences

Dataset: **FPI-Det** by J. Gao et al. (MIT licence), <https://github.com/KvCgRv/FPI-Det>, arXiv:2509.09111.
Detector: Ultralytics YOLO26 (AGPL-3.0). MobileNetV2 weights: ImageNet, via Keras.
