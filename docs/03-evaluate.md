# Phase 3 — Rules, Tuning, and Evaluation

Read `CLAUDE.md` and `docs/PROGRESS.md` first. Requires `models/best.pt` and `data/fpi_yolo/`.
**Syllabus link:** Unit III (object detection), Unit IV (evaluation: accuracy, IoU, mAP),
Unit V (activity recognition: deciding *what the person is doing* from detections).

This phase produces the main results tables. It runs locally on CPU.

## Part A — `ml/rules.py` (usage decision)
Pure functions, no I/O except loading `shared/rules.json`. Input: detections as
`list[{cls: "head"|"phone", box: [x1,y1,x2,y2], conf}]` in pixels, plus the image size.

- `filter_dets(dets, cfg)`: per-class confidence thresholds.
- `pick_user_head(heads, mode)`: `"largest"` (app: the person nearest the webcam) or `"all"` (evaluation).
- `classify_person(head, phones, cfg) -> "ON_CALL" | "ON_PHONE" | "FOCUSED"`:
  - Let `hw, hh` = head width/height and `hc` = head center.
  - **ON_CALL** if any phone box intersects the head box expanded by `call.head_expand × hw`
    horizontally and `× hh` vertically.
  - **ON_PHONE** if any phone center `pc` satisfies `pc.y > hc.y`,
    `|pc.x − hc.x| ≤ max_dx_head_widths × hw`, and `pc.y − hc.y ≤ max_dy_head_heights × hh`.
  - Otherwise **FOCUSED**. ON_CALL wins if both match.
  - Each phone goes to at most one head: the nearest, by center distance divided by head height.
- `classify_image(dets, cfg, mode="all") -> {"using": bool, "persons": [...]}`: the image is "using" if
  any person is ON_CALL or ON_PHONE.
- `ml/tests/test_rules.py` (pytest) with hand-built boxes covering every branch: multiple heads, a
  phone far away on a desk, and a phone overlapping two heads.

## Part B — Tune / held-out split
`label.csv` is the only usage ground truth. Split its images deterministically (seed 42) into
**30% tune / 70% held-out**. Save the lists to `results/eval/splits/`. The tune split is used to tune
rule thresholds here and to train the MobileNet baseline in Phase 4. Every reported number comes from held-out.

## Part C — `ml/tune_rules.py`
Grid search over `conf.phone`, `conf.head`, `call.head_expand`, `using.max_dx_head_widths`, and
`using.max_dy_head_heights` (small grids, ~200–400 combos). Run the model **once** at `conf=0.05`, cache
the detections, then sweep thresholds in pure Python. Objective: F1 of "using" on the tune split.
Write the best values back to `shared/rules.json`, save the full sweep to `results/eval/tuning.csv`,
and save a sensitivity plot for the two most influential parameters.

## Part D — `ml/evaluate.py`
`python ml/evaluate.py` regenerates everything in `results/eval/`.

**1. Detection (test split of `data/fpi_yolo`)**
- Models: `yolo26n.pt` (COCO pretrained, class `cell phone` mapped → `phone`) vs `models/best.pt`.
- Phone **mAP50 and mAP50-95** for both (`torchmetrics.detection.MeanAveragePrecision`, phone class only).
- Phone **mean IoU** of correctly detected phones (IoU of each true-positive box with its matched
  ground-truth box at IoU ≥ 0.5) for both models. The syllabus names IoU explicitly, so report it on its own.
- Head mAP50 / mAP50-95 / mean IoU for the fine-tuned model only (COCO has no head class; say so).
- Phone mAP50 by box size (small / medium / large, COCO area thresholds) for both models.
- Outputs: `detection.json`, `detection_by_size.png`.

**2. Usage classification (held-out split of label.csv)**
| Method | Detector | Decision |
|---|---|---|
| A-coco | COCO yolo26n | any phone detected → using |
| A-ft | fine-tuned | any phone detected → using |
| A2-authors | fine-tuned | the dataset authors' `simple_classify_images.py` logic |
| B-rules | fine-tuned | `rules.classify_image(mode="all")` |

(Phase 4 adds row C-mobilenet.)
- A2: read `third_party/FPI-Det/classification_task/simple_classify_images.py` and feed it our
  detections in the format it expects. **Don't change its decision logic.** If it can't be adapted,
  document why in PROGRESS.md and drop the row.
- Metrics per method: **accuracy** (headline, as the syllabus names it), plus precision, recall, F1,
  and a confusion matrix. Also run the authors' `calculate_metrics.py` on each method's predictions.
- If `label.csv` has behavior categories (e.g. Calling / Using), report B-rules recall per category.
- Outputs: `classification.json`, `confusion_<method>.png`, `summary.md` (clean markdown tables).

**3. Failure analysis**
For B-rules, save `failures_fp.png` and `failures_fn.png`: 3×4 grids of the highest-confidence false
positives and false negatives, with boxes and predicted states drawn. Add 3–5 plain-language bullet
observations to `summary.md`.

**4. Speed**
CPU latency (ms/image, mean of 200 test images after 20 warm-up runs) for both models at 640.
Record the CPU model name.

## Done when
- [x] `pytest ml/tests` passes
- [x] `shared/rules.json` holds the tuned values, tuned on the tune split only
- [x] `python ml/evaluate.py` regenerates all of `results/eval/` from scratch
- [x] `summary.md` has detection (mAP + IoU), classification (accuracy first), and speed tables, plus failure notes
- [x] Sanity check: if B-rules does **not** beat A-ft on accuracy and F1, stop and report to the user with the failure grids before going further (gate failed: B-rules 0.873/0.852 vs A-ft 0.876/0.854; reported, user chose to accept and report)
- [x] PROGRESS.md updated with the headline numbers
