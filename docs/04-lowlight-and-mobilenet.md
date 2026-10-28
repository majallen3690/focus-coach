# Phase 4 — Low-Light Enhancement Experiment + MobileNet Baseline

Read `CLAUDE.md` and `docs/PROGRESS.md` first. Requires Phase 3 to be done (tuned `rules.json`,
splits in `results/eval/splits/`).
**Syllabus link:** Unit II (histogram equalization, filtering; Practical 2) and Unit IV (transfer
learning with MobileNet; Practical 8).

Two independent experiments. Both add rows to `results/eval/summary.md`.

---

## Part A — Does image enhancement help in low light? (Unit II)

### Why
Webcams in dim rooms give dark, noisy frames. Classic enhancement may help the detector see phones.
We **measure** whether it does, rather than assuming it.

### `ml/enhance.py` (pure functions, reused by the app)
- `to_gray_mean(img) -> float`: mean brightness of the grayscale image (0–255)
- `global_he(img)`: histogram equalization on the luminance channel (convert BGR → YCrCb, equalize Y, convert back)
- `clahe(img, clip_limit, tile_grid)`: CLAHE on the L channel of LAB
- `smooth(img, ksize)`: Gaussian blur, for noise
- `enhance(img, method, params)`: dispatcher for `"none" | "global_he" | "clahe" | "clahe_smooth"`
- `ml/tests/test_enhance.py`: shape and dtype preserved; an all-black image doesn't crash;
  `"none"` returns the input unchanged.

### Building the low-light test set — `ml/make_lowlight.py`
From the FPI-Det **test** split, generate darkened copies into `data/lowlight/{dim,dark}/`:
- **dim:** brightness × 0.45, gamma 1.8, Gaussian noise σ = 4
- **dark:** brightness × 0.25, gamma 2.4, Gaussian noise σ = 8
Seeded and deterministic. Labels are unchanged, since darkening doesn't move boxes. Also build a
**real-dark subset**: the 15% of original test images with the lowest mean brightness, so the
synthetic result can be checked against real dark photos.

### `ml/lowlight_experiment.py`
1. Pick CLAHE `clip_limit ∈ {2, 3, 4}` and `tile_grid ∈ {(8,8), (16,16)}` on the **val** split's darkened
   copies (generate them the same way). Choose by phone mAP50.
2. Evaluate the fine-tuned detector on **test**: conditions {original, dim, dark, real-dark} ×
   enhancement {none, global_he, clahe, clahe_smooth}. Report phone mAP50 and head mAP50 per cell.
3. Run B-rules usage accuracy on the held-out label.csv images under {original, dark} × {none, best method}.
4. Figures for `results/lowlight/`:
   - `enhancement_examples.png`: one image as original / dark / dark+global HE / dark+CLAHE /
     dark+CLAHE+smooth, each with its grayscale histogram underneath (the classic Unit II figure)
   - `lowlight_map.png`: grouped bar chart of mAP50 by condition and method
5. **Decide the app setting from evidence:** write to `shared/rules.json`:
   ```json
   "app": { "low_light_threshold": <brightness value>, "enhance_method": "<best or none>",
            "enhance_params": { ... } }
   ```
   Pick the threshold from the brightness distribution of dim/dark frames. If no method beats "none"
   on dark images, set `"enhance_method": "none"` and say so plainly in `summary.md`. A negative result
   is a valid finding: deep models trained with brightness augmentation are often already robust.

---

## Part B — MobileNetV2 transfer-learning baseline (Unit IV)

### Why
Compare our "detect, then reason with rules" approach against the textbook alternative: a CNN
classifier that looks at the whole image and directly predicts "using phone / not using".

### `ml/mobilenet_baseline.py` (Keras / TensorFlow)
- Data: the **tune** split of label.csv (internal 80/20 train/val, stratified, seed 42). Evaluate on
  **held-out** only.
- Model: `keras.applications.MobileNetV2` with ImageNet weights, `include_top=False`, input 224×224,
  then GlobalAveragePooling → Dropout(0.3) → Dense(1, sigmoid). Use the matching `preprocess_input`.
- Augmentation (Keras preprocessing layers): horizontal flip, small rotation (±5°), brightness/contrast jitter.
- **Stage 1 (feature extraction):** freeze the backbone, Adam lr 1e-3, up to 15 epochs, early stopping on val loss.
- **Stage 2 (fine-tuning):** unfreeze the top ~30 layers, Adam lr 1e-5, up to 10 epochs, early stopping.
- Use class weights if the classes are imbalanced (report the ratio).
- Choose the decision threshold on the internal val split (maximize accuracy). Keep 0.5 if it's within 0.02.
- Save `models/mobilenet_baseline.keras` and `results/mobilenet/history.json`.
- Plots: `training_curves.png` (loss + accuracy, both stages, with a vertical line where fine-tuning starts).
- Evaluate on held-out with the same metrics as Phase 3 (accuracy first, then precision, recall, F1,
  confusion matrix). Add **C-mobilenet** to the classification table in `summary.md`.
- Save `results/mobilenet/examples.png`: 8 held-out images where MobileNet and B-rules disagree, with both
  predictions shown. Add 3 plain-language bullets on why they differ (likely: the whole image is shrunk
  to 224 px, so small phones in wide scenes nearly disappear, while the detector looks at full resolution).

### Environment
TensorFlow runs on CPU on Windows; with a frozen backbone and ~1–1.5k images this is minutes, not hours.
If TensorFlow and PyTorch dependencies conflict in `.venv`, create `.venv-tf` just for this script and
document it in the README. If CPU training takes > 30 minutes, add a Colab cell to `train_colab.ipynb`
instead.

---

## Done when
- [x] `pytest ml/tests` passes, including `test_enhance.py`
- [x] `results/lowlight/` has both figures; the low-light table is in `summary.md`
- [x] `shared/rules.json` has an `app` section chosen from the experiment, not guessed
- [x] `models/mobilenet_baseline.keras` exists; training curves saved; C-mobilenet row in `summary.md`
- [x] The final classification table compares A-coco, A-ft, A2-authors, B-rules, and C-mobilenet on the same held-out images
- [x] PROGRESS.md updated with both findings in plain words
