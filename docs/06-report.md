# Phase 6 — README, Figures, Report Draft, Viva Prep

Read `CLAUDE.md`, `docs/PROGRESS.md`, and `results/eval/summary.md` first. Every number in this phase
must come from files in `results/`. **Never type a metric by hand.** If a number is missing,
regenerate it or flag it.

## Step 1 — `report/build_report.py`
Reads `results/**/*.json` and fills tables into `report/report.md` from `report/template.md`
(placeholders). Re-running after any re-evaluation updates the report.

## Step 2 — Figures → `report/figures/` (200 dpi, consistent fonts)
1. **Pipeline diagram** (Mermaid in markdown + exported PNG): webcam → brightness check → CLAHE (if dark)
   → YOLO26n → head/phone boxes → rules → time smoothing → events → session report
2. **Rule geometry**: a sample image with a head box, the expanded ON_CALL zone, and the ON_PHONE
   reach zone, drawn using the real tuned values from `rules.json`
3. Dataset samples grid + box-size histogram (Phase 1)
4. Training curves + COCO-vs-fine-tuned before/after examples (Phase 2)
5. Phone mAP50 by box size: COCO vs fine-tuned (Phase 3)
6. Confusion matrices: A-ft, B-rules, C-mobilenet side by side
7. Failure grids, FP and FN (Phase 3)
8. Enhancement examples with histograms + low-light mAP chart (Phase 4)
9. MobileNet training curves + MobileNet-vs-rules disagreement examples (Phase 4)
10. A sample session `report.png` from the app (Phase 5)

## Step 3 — `report/report.md`
Plain, clear academic English. No hype.
1. **Abstract** (≤150 words)
2. **Introduction**: phone distraction while studying; why "phone visible" ≠ "phone in use"; aims
3. **Syllabus mapping**: a table mapping each part of the project to Units I–V, the practicals
   (2, 6, 8, 11), and CO1–CO5, using the table in `CLAUDE.md`
4. **Related work**: FPI-Det (Gao et al., ICASSP 2026); YOLO object detectors; MobileNet;
   histogram equalization / CLAHE. **Only cite verifiable sources. Leave `[CITE]` markers where the user must add a reference.**
5. **Method**
   - 5.1 Image acquisition and preprocessing (OpenCV, resizing, brightness check)
   - 5.2 Low-light enhancement (global HE, CLAHE, Gaussian smoothing)
   - 5.3 Object detection with YOLO26n; transfer learning and fine-tuning setup
   - 5.4 Usage rules (with the geometry figure)
   - 5.5 Time smoothing as simple activity recognition
   - 5.6 Baseline: MobileNetV2 transfer-learning classifier
6. **Experiments**: data splits (incl. the 30/70 tune/held-out split of label.csv and why), metrics
   (accuracy, IoU, mAP; precision/recall/F1 explained in one line each), hardware
7. **Results**: detection (mAP, IoU, by size), usage classification (A-coco, A-ft, A2-authors, B-rules,
   C-mobilenet), low-light study, speed, failure analysis
8. **Application: Focus Coach**: screenshots (placeholders for the user), what a session report shows
9. **Limitations**: FPI-Det is mostly surveillance-angle images while the app uses a frontal webcam;
   time smoothing is demonstrated but not measured (no labeled video); phones held below the camera's
   view can't be seen; single-user assumption (largest head)
10. **Ethics & privacy**: self-monitoring by consent only; frames never stored by default
11. **Future work**: a labeled webcam video set for measuring the smoothing; pose estimation (Unit V)
    to catch "looking down at a phone below the frame"
12. **References**

## Step 4 — `README.md` (repo root)
Pitch, screenshot placeholder, headline results table (auto-filled from `summary.md`), quickstart
(venv → evaluate → run the app), Colab training via `docs/COLAB.md`, key controls (S/E/D/C/Q),
repo layout, dataset credit and license (FPI-Det, MIT), privacy note.

## Step 5 — `report/viva.md`
- **3-minute demo script:** start session → FOCUSED → quick glance (no event) → pick up phone (event)
  → hold to ear (ON_CALL) → turn off the room light (CLAHE kicks in) → end session → walk through the
  report. Backup: `--source demo.mp4`.
- **12 likely examiner questions**, each with a 2–3 sentence plain answer grounded in the project's
  actual numbers. Must include:
  - What is transfer learning, and why fine-tune instead of using COCO as-is?
  - What do mAP and IoU measure? What are our numbers?
  - Why rules after detection instead of one classifier? (Use the MobileNet comparison.)
  - What does CLAHE do differently from global histogram equalization? Did it help?
  - How does YOLO detect objects in one pass? Why doesn't YOLO26 need an NMS step?
  - Why does the 1.5-second smoothing exist?
  - What fails, and why? (Use the failure grids.)
  - How was data leakage avoided? (Tune vs held-out split.)
- **Out-of-syllabus items** used (e.g. time smoothing, precision/recall/F1), each with a one-line
  plain explanation the user can say out loud.

## Done when
- [x] `python report/build_report.py` produces `report/report.md` with every table filled from `results/`
- [x] All 10 figures exist in `report/figures/`
- [ ] README quickstart works from a fresh clone (data and model downloads documented) (documented, NOT verified end to end: needs the dataset download and a trained model)
- [x] `report/viva.md` written with the demo script, 12 Q&As, and the out-of-syllabus list
- [x] All `[CITE]` and screenshot placeholders listed at the end of PROGRESS.md for the user
