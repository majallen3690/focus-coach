# Progress log

## Phase 0: Scaffold (2026-10-26)

**Built:** repo layout, `.gitignore`, `shared/rules.json` (starting values, not tuned yet),
`docs/RULES.md` (the rule spec), phase docs 01–06, README skeleton, Python 3.11 venv.
`third_party/FPI-Det` is a shallow clone of https://github.com/KvCgRv/FPI-Det.

**What the FPI-Det repo actually contains (inspected, not assumed):**
- `annotations/instances_train2017.json`: COCO format, 18,800 images, 33,350 boxes
  (cat 0: 8,358, cat 1: 24,992).
- `annotations/instances_val2017.json`: 1,730 images, 2,113 boxes (cat 0: 795, cat 1: 1,318).
- Categories are named `class_0` and `class_1` in the JSON. `Phone.yaml` maps `0: phone`, `1: face`,
  and the README calls the second class "head". The counts back this up (heads outnumber phones).
  **We use 0 = phone, 1 = head.**
- `bbox` is COCO `[x, y, w, h]` in pixels, floats. Sample images are small crops
  (116×270, 324×456): these are **person crops from surveillance footage**, not full scenes.
- `classification_task/label.csv`: 4,079 rows, columns `image_name,class_id`. `class_id` is a float
  (`0.0`/`1.0`). 2,425 are `0` and 1,654 are `1`.
  **Correction (Phase 1):** I first wrote that 0 = using phone, copying the docstring of
  `simple_classify_images.py`. The boxes show the opposite: 1 = using phone (see the Phase 1
  entry). That docstring is wrong.
- The other CSVs (`8n.csv`, `11n.csv`, `result.csv`, …) are example predictions from the paper's
  baselines.
- `phone_train_yolo.rar` (3.4 MB) is a vendored copy of the Ultralytics repo plus training scripts.
  It is code, not data. We don't use it.
- **Images are not in the repo.** They come from the Google Drive or Baidu link in the README.
- README totals: 22,879 images, 29,279 heads, 10,255 phones. COCO train + val = 20,530 images, so
  about 2.3k images (likely the test or classification images) are only in the download.
  To verify in Phase 1.
- `Phone.yaml` mentions train/val/test = 6,471/548/1,610. That doesn't match the COCO JSON, so it's
  probably from an older version. The real split comes from the download.

**Open issues**
1. **Domain gap:** the training data is surveillance crops, but the app sees a frontal laptop webcam
   at arm's length. Expect lower accuracy live. Phase 5 needs a small manual webcam check.
2. **Binary labels only:** `label.csv` can't tell ON_CALL apart from ON_PHONE, so only the binary
   "using phone" decision gets a score.
3. ~~Possible leakage~~ Checked in Phase 1: none (see below).
4. **Licences:** Ultralytics is AGPL-3.0. FPI-Det is MIT. Mention both in the report.

## Phase 1: Dataset prep (2026-10-26)

**Built:** `ml/prepare_dataset.py` (one command, idempotent, ~20 min on OneDrive), producing
`data/fpi_yolo/` (+ `fpi.yaml`, `usage_labels.csv`), `data/fpi_yolo.zip` (1.17 GB, for Colab),
`results/dataset_stats.json`, `results/splits.json`, `results/label_check/{train,val,test}.jpg`.

**What the real download looks like** (the Phase 0 guesses about COCO conversion and a missing test
split were wrong):
- `data/raw/reorganized_dataset/{train,val,test}/{images,labels}`, already YOLO format, class
  `0` = phone, `1` = head. Images: 18,800 / 1,730 / 2,349. Train and val are the COCO images;
  test is not in the COCO JSON.
- The raw splits are used unchanged: det_train = train, det_val = val, det_test = test.
- **`label.csv` = val + test exactly** (1,730 + 2,349 = 4,079, no other images), so there is no leakage
  into train. `usage_val` = val rows, `usage_test` = test rows.
- **`label.csv` polarity: `class_id` 1 = using phone.** In val and test, 97.3% of class-1 images have
  a phone box vs 11.8% of class-0. The docstring in the repo's `simple_classify_images.py` says the
  opposite and is wrong. The script asserts this on every run.

**Cleaning (counts from `results/dataset_stats.json`):**
- 242 label files had several boxes on one line (missing newlines); split by token groups of 5.
  This recovers ~200 phone boxes the COCO JSON doesn't have: train has 8,566 phones, where the
  COCO JSON and a naive line count give 8,358.
- 1,122 boxes extended past the image edge by more than 0.0001 (not float noise); clipped.
- 4 degenerate boxes dropped (e.g. `1 -0.892 0.035 0 0`), 3 in train and 1 in test.

**Final numbers**

| split | images | head boxes | phone boxes | images with no boxes |
|---|---|---|---|---|
| train | 18,800 | 24,991 | 8,566 | 48 |
| val | 1,730 | 1,318 | 809 | 387 |
| test | 2,349 | 2,967 | 1,132 | 25 |

Usage labels (using phone = 1): val 691 of 1,730 (40%), test 963 of 2,349 (41%).

**Checked:** an independent script confirmed image/label pairing, valid boxes, counts equal to the
stats file, zip contents equal to the dataset folder, and that Ultralytics' `check_det_dataset`
loads `fpi.yaml`. Contact sheets of 10 random images per split show boxes in the right places.
A second run gave the same counts.

**Open issues for later phases**
1. **Faces are masked with grey bars in the images** (anonymisation). The detector will learn heads
   with a grey bar across the eyes; a live webcam face has none. Likely to hurt head recall live.
   Check in Phase 5; mention in the report.
2. **Phones without a head box.** In val, 292 of the 691 "using phone" images (42%) have a phone box
   but no head box; in test only 2 of 963. The rules need a head, so these would be scored
   "not using" unless Phase 3 adds a rule for a phone with no visible head. Val and test differ a lot
   here, so tune carefully: a rule that helps on val may not help on test.
3. **Some label noise:** phones visible but unlabelled in a few contact-sheet images.
4. The dataset is person crops (median 250×500 px; median head is ~5% of the image area, median
   phone ~1.3%), not full webcam frames. Object scale in the app is untested.

## Phase 2 and 2b: Training and catch-up (2026-10-28)

**Training ran on Colab (T4) with the first version of `ml/train_colab.ipynb`**, which was written before
the new Phase 2 spec. Phase 2b then regenerated the outputs that run did not produce (no retraining).

**What was trained** (read from `results/train/args.yaml` and `train_summary.json`, not from memory)
- YOLO26n from COCO weights, all layers fine-tuned, `imgsz` 640, seed 42, `deterministic=True`, default
  Ultralytics augmentation (mosaic on for the first 10 epochs, off for the last 10), `cache="disk"`.
- **20 epochs**, batch **32**, patience 15 (never triggered). Tesla T4, **1.50 h** (about 4-5 min per epoch).
- Colab versions: ultralytics 8.4.190, torch 2.11.0+cu130. Optimizer was left on `auto`; which one
  Ultralytics picked is not saved anywhere.
- **Differs from the current spec** (`docs/02-train-colab.md`: `EPOCHS = 60`, AutoBatch). 02b assumed
  "same settings"; that is true except for these two. The model was not retrained.

**Results: `best.pt` on the val split** (recomputed on CPU; matches the Colab-recorded numbers to 0.0012)

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| phone | 0.891 | 0.825 | 0.894 | 0.630 |
| head | 0.550 | 0.894 | 0.548 | 0.345 |
| all | 0.721 | 0.859 | 0.721 | 0.487 |

- **Curves** (`results/train/results.png`): val mAP50 rose from 0.596 (epoch 1) to 0.723 (epoch 20) and
  mAP50-95 from 0.301 to 0.488. Still rising slowly at the end (about +0.002 mAP50 per epoch over the
  last four), no sign of overfitting. More epochs would help a little; 20 is a reasonable stopping point.
- **Before/after** (`results/train/before_after.png`): on 6 val images chosen for having small phones, COCO
  `yolo26n` found a phone in 3 of 6, the fine-tuned model in 6 of 6 (both at confidence 0.25). This is an
  illustration, not a metric; Phase 3 measures it properly.
- **Head precision is low (0.55) while recall is high (0.89).** Hypothesis, not yet checked: many val
  images have heads that are not labelled (387 of 1,730 val images have no boxes at all; Phase 1), so
  correct detections count as false positives. Test: if head precision is much higher on the test split
  (only 25 of 2,349 images have no boxes), label gaps explain it. Do this in Phase 3.

**Outputs** in `results/train/`: `val_per_class.json`, `before_after.png`, `train_summary.json`,
`results.csv`, `args.yaml`, training plots (`results.png`, PR/F1/P/R curves, confusion matrices;
`val_*` copies come from the notebook's own validation), `phase2_summary.json` (what the first notebook
recorded). `models/best.pt` (5.4 MB) and `models/yolo26n.pt` (COCO weights) are in `models/`, not committed.

**Code added:** `ml/phase2_catchup.py` (one command, about 1 min on CPU), `ml/train_report.py`
(per-class table, before/after figure, data preview, summary; shared by the script and the notebook),
`docs/COLAB.md`. `ml/train_colab.ipynb` rewritten to the current Phase 2 spec.

**Notebook status (not run on Colab; per 02b it must not be run here).** Tested locally on CPU against the
real Colab run: the data-preview, train (finished run is detected and skipped), validate, before/after and
save cells. **Not tested:** Drive mount, repo clone, the download-and-build branch (`gdown` + 7z; the
archive format of the authors' download was never checked), `pip install -r requirements.txt` on Colab,
and resuming an interrupted run (rests on Ultralytics' documented `resume=True`).
Found while testing: calling `resume=True` on a **finished** run does not fail, it silently starts a new
default training. The notebook therefore checks `epoch == -1` in `last.pt` first (verified on two finished
checkpoints; an unfinished one keeps its real epoch per the Ultralytics source).
- `REPO_URL` in the notebook is a placeholder; the user must set it. The repo has no git remote yet.
- `DRIVE_DIR` defaults to `MyDrive/focus-coach` (hyphen); the first run used `MyDrive/focus_coach`
  (underscore), so an old `fpi_yolo.zip` there will not be found unless it is moved or `DRIVE_DIR` is changed.

**Fixed along the way:** `train_report.py` first forced matplotlib's `Agg` backend on import, which would
have made the notebook's `plt.show()` display nothing on Colab. Now only the headless script sets it.
A local test of `resume=True` left a stray `runs/` folder in the repo root; deleted.

**Things to reconcile after the CLAUDE.md and docs rewrite (decided later, not changed now)**
1. **Rules and config:** `shared/rules.json` and `docs/RULES.md` are Phase 0 drafts with different key
   names. Phase 3 uses `conf.*`, `call.*`, `using.*`, `smoothing.window_ms / enter_ratio / away_ms`, `app`.
   Both get rewritten in Phase 3.
2. **Splits:** Phase 1 set usage_val = val and usage_test = test (`results/splits.json`). Phase 3 now asks for
   a seeded 30/70 tune/held-out split of `label.csv`. `label.csv` is exactly val + test, so about 42% of it is val
   images that were used to pick `best.pt`: a small optimistic bias on held-out numbers; say so in the report.
3. **Phase 1 layout:** CLAUDE.md lists `ml/inspect_dataset.py` and `ml/dataset_stats.py`; Phase 1 did
   inspection by hand and puts the stats in `prepare_dataset.py`. Left as is.
4. **Syllabus scope:** `onnx` and `onnxruntime` removed from `requirements.txt` (Phase 7 only). The
   `.venv` still has them installed; harmless.
5. **Label data:** 2 val images contain an exact duplicate box (Ultralytics drops them at load time:
   val has 2,125 instances, my stats say 2,127). Harmless; `prepare_dataset.py` does not dedupe.

## Phase 3: rules, tuning, evaluation (done; sanity gate failed, user chose to accept and report)

**Built:** `ml/rules.py` (28 passing tests), `ml/splits.py` (seed 42, tune 1,224 / held-out 2,855, stratified by
val/test and label), `ml/detections.py` (detector run once, cached in `data/cache/`), `ml/tune_rules.py`,
`ml/evaluate.py` with `eval_detection.py`, `eval_usage.py`, `usage_metrics.py`. All outputs are in `results/eval/`
(`summary.md` has every table). `torchmetrics==1.9.0` and `pycocotools==2.0.11` added to `requirements.txt`.

**Tuned on the tune split only** (`shared/rules.json`): phone conf 0.2, head conf 0.1, head_expand 0.1,
max_dx 3.0 head-widths, max_dy 2.0 head-heights. The first 405-combination grid had its best value on the edge for
three parameters, so it was widened to 540 combinations (tune split only; held-out was not looked at before that).
The best value still sits on the edge for head conf and max_dx, but the gain from widening was tiny (tune F1 0.845 to 0.847).

**Headline numbers (held-out, n=2,855; our metrics match the authors' `calculate_metrics.py` to 4 decimals):**

| Method | Accuracy | F1 |
|---|---|---|
| A-coco (any phone, COCO model) | 0.679 | 0.632 |
| A-ft (any phone, fine-tuned) | 0.876 | 0.854 |
| A2-authors (their logic on our boxes) | 0.862 | 0.843 |
| B-rules | 0.873 | 0.852 |

Detection (test split): phone mAP50 0.313 (COCO) to 0.869 (fine-tuned); mAP50-95 0.103 to 0.604; head mAP50 0.954.
Phone mAP50 by size, fine-tuned: small 0.79, medium 0.89, large 0.95. CPU (i7-13700HX): about 30 ms per image
(fine-tuned, 640 px, batch 1; the very first run measured 82 ms and later runs 31 to 32 ms; the first was probably slowed by other load, not verified).

**Sanity gate failed:** B-rules does not beat A-ft (accuracy 0.873 vs 0.876, F1 0.852 vs 0.854). The gap is tiny,
but per `docs/03-evaluate.md` work stopped for the user's decision. The user chose to accept and report it as is;
nothing was tuned further after seeing held-out.
Failure grids and notes are in `results/eval/` (`failures_fp.png`, `failures_fn.png`, `failure_notes.md`).
Main reading: false positives and false negatives are the same kind of scene (phone held low or to the side) with
opposite labels, so part of the error is probably label noise; 93 of 116 misses have no phone detected at all.

**Reconciled:** `docs/RULES.md` rewritten to match `ml/rules.py`; README Phase 3 row and results updated;
`docs/03-evaluate.md` boxes ticked. `results/splits.json` (Phase 1) is superseded by `results/eval/splits/`.

**For the report and `report/viva.md`:** say plainly that the rules tie the "any phone" baseline on FPI-Det and
why (label noise, detector misses, no head box on about 18% of "using" images, and the rules need a head).
Their value is the ON_CALL / ON_PHONE / FOCUSED states for the app. `torchmetrics` (listed in CLAUDE.md) and its
`pycocotools` backend are used only to compute mAP, the Unit IV metric; no other out-of-syllabus technique was added.
**Open for later:** the `smoothing` values are untuned starting values (Phase 5 uses them on video); thresholds were
tuned on surveillance-style photos, not laptop-webcam frames.

## Phase 4: low-light experiment and MobileNet baseline (done)

**Built:** `ml/enhance.py` (+ tests), `ml/make_lowlight.py` (dim/dark copies of the 2,349 test images in `data/lowlight/`,
real-dark subset of 352), `ml/lowlight_experiment.py` + `lowlight_figures.py`, `ml/mobilenet_baseline.py` (TensorFlow, own
`.venv-tf`, pinned in `requirements-tf.txt`; main `.venv` untouched), `ml/eval_mobilenet.py`, `ml/phase4_summary.py`.
`evaluate.py` now adds the low-light and MobileNet sections and the C-mobilenet row to `results/eval/summary.md`.
48 tests pass.

**Part A finding (detector phone mAP50, test split).** The "dim" copies (mean brightness about 16) are the only place
enhancement helps: none 0.699, CLAHE + blur 0.737 (head 0.837 to 0.864). Global HE (0.600) and plain CLAHE (0.672) hurt,
because they amplify the noise. The "dark" copies (mean brightness about 4) are too dark: 0.00 for every method, and B-rules
then calls everything "not using" (accuracy 0.594 = the share of "not using"). On real dark photos (brightness up to 75)
and on original images, "none" is best or tied, so enhancement is not needed there.
**App setting chosen from this** (`shared/rules.json` `app`): `clahe_smooth`, clip 2.0, tiles 8x8, blur 3, used when mean
brightness is below 29 (the 95th percentile of the dim frames). Method and CLAHE settings were chosen on a 400-image val
sample, then confirmed on test. A first version of the script set the threshold at 75 (the brightest real-dark photo); that
would have switched enhancement on where it slightly hurts, so the rule was changed before any further use.
For Phase 5: the app should warn "too dark" below roughly brightness 8 instead of showing detections.

**Part B finding (held-out, n=2,855).** MobileNetV2 transfer learning: accuracy **0.736**, F1 0.706, far below B-rules (0.873 /
0.852). Trained on the tune split only (979 train / 245 validation, class ratio 1.47:1, class weights used), 15 + 10 epochs
(both hit the epoch cap with validation still improving slowly), threshold 0.5 (best on validation 0.65, gain under 0.02, so
0.5 kept). About 6 minutes on this CPU, so no Colab cell was needed. The two methods disagree on 778 images; MobileNet alone is
right on 193, B-rules alone on 585. Likely reasons are in `results/eval/summary.md` (small phones vanish at 224 px; much less
training signal).

**Scope notes (CLAUDE.md rule 5):** `scikit-learn` (already in requirements) is used only for `train_test_split`; no other
out-of-syllabus technique. Keras preprocessing layers (flip, rotation, brightness, contrast) are the augmentation, Unit IV.
**Caveats:** darkening is a recipe, not a real dim room; the val sample used for selection is darkened by the same recipe.
`models/mobilenet_baseline.keras` is gitignored with the other models. Training is seeded but TensorFlow on CPU is not
guaranteed bit-for-bit repeatable.

## Phase 5: Focus Coach desktop app (code written and tested without a person in front of the camera; live self-test skipped at the user's request)

**Built:** `ml/smoothing.py` (time-weighted window, AWAY, events), `app/capture.py` (threaded latest-frame reader, DirectShow
webcam or video file), `app/session.py`, `app/hud.py`, `app/report.py`, `app/pipeline.py` (enhancement, YOLO, rules for one frame),
`app/focus_coach.py` (main loop and keys), `app/benchmark.py`, `app/make_demo_clip.md`. `shared/rules.json` `app` got `imgsz: 640`.
76 tests pass (13 smoothing, 9 session/report, 6 pipeline).

**Decisions to know about:**
- A state is committed when it fills 60% of the last 1.5 s *in time*, so it appears about 0.9 s after it really starts and ends
  about 0.9 s after it stops; the two delays cancel in durations. A one-frame state after a camera stall counts for at most 0.5 s.
- AWAY (no head for more than 3 s) is dated back to the last frame with a head and is not phone time. Phone to call is one
  pickup, two events. Pause closes any open event and the state restarts as FOCUSED after resume.
- **C** cycles enhancement auto / on / off (the doc said "toggle"). `--autostart` and `--no-display` were added for scripted runs.

**Checked without a person:** a scripted 38 s clip built from real held-out photos (focused, 5 s phone, 0.5 s glance, 5 s call,
6 s black frame). The app logged 2 pickups (phone, call), ignored the glance, showed AWAY for 6 s, and wrote `events.csv`,
`summary.json`, `report.png`. (A video-codec blend between two photos made one extra 1 s ON_CALL event right after the phone
phase; that is an artefact of the test clip, not of the app.) The HUD was rendered to images and looked right.
Webcam reader tested alone: opens, 1280 x 720, no dropped frames, nothing saved.
**Speed:** whole pipeline (no camera, no window) 26.5 ms per frame = 37.7 FPS at imgsz 640, 22.4 ms = 44.7 FPS at 480, on the
i7-13700HX. **But the webcam itself delivers only 10 FPS in the current room light** (brightness about 11 of 255, auto-exposure
slows the camera; MJPG made no difference), so live FPS will show about 10, and the room will count as low light (below 29), so
CLAHE will switch on.

**Not verified:** the on-screen window and key handling (S, E, D, C, Q), opening the report window, `--save-video`, and the
5-minute self-test below. **Risks for the self-test:** `conf.head` is 0.1 (tuned on photos); on a webcam a false large head box could
be picked as the user, and phone/head thresholds were tuned on surveillance-style photos. If the label flickers or misses, adjust
`conf.*`, `using.*` in `shared/rules.json` (the single source of truth) and tell me what you saw.

**Self-test waived:** on 2026-10-29 the user said the 5-minute self-test is not needed and asked to continue. So the live window,
keys, report window and live FPS are **not verified**; the report (Phase 6) must not claim live results. Treat any issue found
during the viva rehearsal as a Phase 5 bug.

## Phase 6: README, figures, report draft, viva prep (done, with one unverified item)

**Built:** `report/build_report.py` (+ `ml/report_values.py`) fills `report/report.md` (4,941 words, abstract 140 words),
`report/viva.md` and the README results block from `results/`; templates are `report/template.md` and `report/viva_template.md`.
The observation notes (`results/eval/failure_notes.md`, `results/lowlight/notes.md`, `results/mobilenet/notes.md`) now use
`{{placeholders}}` too, so no metric is typed by hand. A missing value stops the build; "narrative checks" warn if a result
changes enough to make a template sentence untrue. `report/make_figures.py` and `report/make_sample_session.py` make the 15
files in `report/figures/` (all 10 requested figures; several have an a/b part). Figure code was set to 200 dpi; the Ultralytics
`results.png` and the Phase 2 before/after sheet keep their original resolution. README rewritten (quickstart, key controls,
layout, credits); `requirements-tf.txt` documented. 76 tests pass.

**Correction found while writing the viva notes:** CLAUDE.md says YOLO26 has "no NMS step". Per the Ultralytics docs it has two
heads: the default predict/val path uses the one-to-many head **with** NMS, and the NMS-free one-to-one head is opt-in
(`nms=False`). I checked our checkpoint: it has both heads and the default path applies NMS. All numbers in this project use the
default (NMS) path; the report and viva say so, and list the NMS-free mode as future work.

**Not verified:** the README quickstart from a truly fresh clone (needs the dataset download and a trained `best.pt`, which are
not in the repository). Figure 10 is from a scripted clip built from real photos, not a real study session, and is labelled so.
The live app (window, keys, live FPS) is still unverified (self-test waived, see Phase 5).

**Placeholders for the user to fill (all in `report/report.md` unless noted):**
1. `[AUTHOR: name and enrolment number]` and `[GUIDE: name]` on the title page.
2. `[CO: fill from the syllabus]` in the syllabus table (Section 3): CLAUDE.md names CO1 to CO5 but not their text.
3. `[CITE: Ultralytics YOLO26 documentation or paper]` in Section 4 and as reference [7].
4. `[CITE: confirm the ICASSP 2026 proceedings details]` under reference [1] (FPI-Det: authors and title checked on arXiv 2509.09111;
   the ICASSP 2026 claim comes from the dataset's own README).
5. `[CITE: course textbook chapter on histogram equalization and filtering]` in Section 4.
6. `[SCREENSHOT 1-3]` in Section 8, and `[SCREENSHOT]` in README.md (live window FOCUSED, ON PHONE with boxes, low light).
References [2] to [6] (YOLO, COCO, MobileNetV2, ImageNet, CLAHE) were written from memory of well-known papers; please check each
against the original before submitting.

## Phase 7: Colab demo with Gradio (optional add-on; built and tested locally, the Colab run is not verified)

Note: CLAUDE.md changed during the project; Phase 7 is now the Gradio demo (`docs/07-colab-demo.md`), and the earlier web-app plan is gone.

**Built:** `app/offline.py` (`process_video`, `run_to_folder`, CLI `python -m app.offline clip.mp4 --out folder/`; video timestamps, so
results do not depend on speed), `demo/gradio_app.py` (`build_demo`, tabs Live / Video / Image, footer numbers read from
`results/eval/classification.json`), `demo/focus_coach_colab.ipynb` (7 code cells), `demo/make_examples.py` and `demo/examples/`
(4 held-out photos, a scripted 38 s clip), `docs/COLAB_DEMO.md`, `requirements-demo.txt` (gradio 6.31.0, installs cleanly next to
the pinned requirements, no changes to numpy or pillow), README section, 3 viva Q&As. Gradio is used only in `demo/`.
Behaviour-neutral changes to Phase 5 code so the demo can reuse it: `FocusCoach` can take a shared pipeline and custom key/start
hints, `hud.draw_footer/draw_tip` take their hint text as arguments, `write_report` can take an explicit folder. 84 tests pass
(2 for offline, 6 for the demo).

**Checked locally:** the Image tab on all four examples through the Gradio API (ON_CALL, FOCUSED, ON_PHONE, and the tricky photo
is called "using" as expected); the Video tab through the API on the scripted clip (H.264 output via ffmpeg, report and downloads);
**consistency:** the Video tab and `python -m app.offline` gave the same events on the same clip (2 pickups; ON_PHONE 5.0 s,
ON_CALL 5.0 s; the 0.5 s glance ignored; start offsets identical); the live handlers driven by a scripted frame sequence at 5
frames/s (1 pickup of 5.0 s, glance ignored; unpaced about 30 frames/s on this CPU, server side only); the page renders and its
footer shows the accuracy from `results/`. The notebook's runtime-check and smoke-test cells were run locally.

**Not verified (needs you):** a fresh Colab T4 run of the notebook (Drive mount, clone, install, share link); the live tab with a
real browser webcam and network (the Gradio client cannot call streaming endpoints, so the live logic was tested by calling the
handlers directly); playing the annotated video in a browser; the live frame rate over the internet. `REPO_URL` in the notebook and
the Colab badge in the README point at majallen3690/focus-coach (private). The scripted clip had a codec artefact (a ghost frame blend between
two photos made a false extra event) until it was encoded from an all-keyframe source; the shipped clip is clean.
Also record your own `demo/examples/demo.mp4` for the viva (see `app/make_demo_clip.md`).
