# Focus Coach — Project Rules for Claude Code

## What this project is
Focus Coach is a Computer Vision mini project on **Mobile Phone Usage Detection**
(Course 71407001002 / 73407001002, Karnavati University, Sem VII).

A laptop webcam watches the user during a study session. A desktop app built with OpenCV detects
when the user picks up their phone and produces a session report (pickups, minutes lost, timeline).

Pipeline:
webcam (OpenCV) → low-light enhancement (CLAHE, if needed) → YOLO26n detects `head` + `phone`
→ geometric rules decide usage state → time smoothing → events → session report.

## Stay inside the syllabus
This project is marked against the course syllabus. Every technique must map to a unit below.
If you need something outside this list, **flag it in `docs/PROGRESS.md` and explain it in plain words
in `report/viva.md`**. Never add it silently.

| Unit / Practical | Where it is used |
|---|---|
| Unit I: CV pipeline, image acquisition, OpenCV basics | `app/` webcam capture, frame pipeline, drawing |
| Unit II: histogram equalization, filtering (Practical 2) | `ml/enhance.py` (global HE, CLAHE, Gaussian smoothing) + low-light experiment |
| Unit III: YOLO object detection (Practical 6) | Fine-tuned YOLO26n for head + phone |
| Unit IV: transfer learning, fine-tuning, MobileNet (Practical 8); metrics: accuracy, IoU, mAP | YOLO fine-tuning; MobileNetV2 baseline in Keras; all evaluation |
| Unit V: activity recognition, real-world case studies | Phone-use states + time smoothing; the Focus Coach application |
| Practical 11: Research Project (CO1–CO5) | The whole project |

**Out of scope for Phases 1–6, do not use:** web frameworks, ONNX or browser inference, Streamlit/Gradio,
Grad-CAM, transformers/ViT, tracking libraries, LLM APIs.

**Exception: Phase 7 (optional Colab demo)** may use Gradio, but only inside `demo/` and `app/offline.py`.
It must reuse the Phase 1–6 model, `ml/rules.py`, `ml/smoothing.py`, and `ml/enhance.py` unchanged. Never change
CV logic to suit the demo.

## Usage states (the core idea)
For the user's head box H and each phone box P:
- **ON_CALL**: P overlaps H expanded by a margin (phone at the ear/face)
- **ON_PHONE**: P is below H's center and within a horizontal/vertical reach of H (phone held in front)
- **FOCUSED**: no phone, or phone outside reach (e.g. lying far away on the desk)
- **AWAY**: no head detected for > 3 s (app only)

A state only counts after it holds for a smoothing window (~1.5 s), so brief glances are ignored.

## Single source of truth for thresholds
All thresholds and app settings live in `shared/rules.json`. Code reads them from there.
**Never hard-code a threshold anywhere else.**

## Repo layout
```
focus-coach/
├── CLAUDE.md
├── docs/                    # phase prompts + PROGRESS.md
├── shared/rules.json        # thresholds + app settings (single source of truth)
├── third_party/FPI-Det/     # git clone of the dataset repo — gitignored
├── data/                    # gitignored: raw/, fpi_yolo/, lowlight/
├── ml/
│   ├── inspect_dataset.py, prepare_dataset.py, dataset_stats.py
│   ├── train_colab.ipynb
│   ├── rules.py, smoothing.py, enhance.py
│   ├── tune_rules.py, evaluate.py
│   ├── lowlight_experiment.py, mobilenet_baseline.py
│   └── tests/
├── app/                     # OpenCV desktop app
│   ├── focus_coach.py, capture.py, hud.py, session.py, report.py
├── models/                  # gitignored: best.pt, mobilenet_baseline.keras
├── results/                 # metrics JSON, plots, failure grids (committed)
├── sessions/                # gitignored: saved session reports
├── report/                  # report draft, figures, viva notes
└── demo/                    # Phase 7 only: Gradio app + Colab notebook (optional add-on)
```

## Environment
- OS: **Windows**, shell: **PowerShell**. Commands must work there.
- Python 3.11 in a venv at `.venv`. Install into the venv only; pin versions in `requirements.txt`.
- Libraries: OpenCV, Ultralytics (PyTorch), TensorFlow/Keras (MobileNet baseline only), NumPy,
  pandas, matplotlib, torchmetrics, pytest.
- Phase 7 only: Gradio (runs in Google Colab).
- YOLO training runs on **Google Colab (T4 GPU)**. The local machine is CPU-only: evaluation,
  MobileNet (frozen backbone), and the app run on CPU.

## Working rules
1. **Use the Context7 MCP** before writing code against Ultralytics, OpenCV, or Keras APIs.
   YOLO26 differs from YOLOv8 (end-to-end output, no NMS step). Don't rely on memory of older APIs.
2. **Inspect, don't assume.** The FPI-Det structure, class names, and `label.csv` format must be read
   from the actual files. Record the findings in `docs/PROGRESS.md`.
3. One phase at a time. Each phase file ends with a **Done when** checklist. Don't start the next phase
   until every box is satisfied. Then append a short entry to `docs/PROGRESS.md`
   (what was built, key numbers, open issues).
4. If blocked by something only the user can do (download, Colab run, camera test), **stop and give
   exact steps**.
5. Never commit `data/`, `models/`, `sessions/`, `third_party/`, or `.venv/`.
6. Evaluation hygiene: tune and train on the **tune** split only; report on the **held-out** split once.
   Never tune on held-out.
7. Small readable modules, type hints, docstrings. No dead code or leftover TODOs in finished phases.
8. Every script runs from the repo root with one command, documented in its docstring and the README.
9. Plain language in all user-facing text, reports, and viva notes.

## Phase order
| # | File | Output |
|---|------|--------|
| 1 | docs/01-dataset-prep.md | YOLO-format dataset + `fpi.yaml`, dataset stats |
| 2 | docs/02-train-colab.md | Colab notebook → `models/best.pt` |
| 2b | docs/02b-phase2-catchup.md | Only if training ran on the earlier notebook: regenerate missing Phase 2 outputs |
| 3 | docs/03-evaluate.md | Rules, tuning, detection + usage-classification results |
| 4 | docs/04-lowlight-and-mobilenet.md | CLAHE low-light experiment + MobileNet transfer-learning baseline |
| 5 | docs/05-opencv-app.md | Focus Coach desktop app (OpenCV) with session reports |
| 6 | docs/06-report.md | README, figures, report draft, viva prep |
| 7 | docs/07-colab-demo.md | Optional: Gradio demo runnable from Colab with a shareable link (start only after Phase 6) |
