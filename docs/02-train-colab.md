# Phase 2 — Fine-tune YOLO26n on FPI-Det (Colab)

Read `CLAUDE.md` and `docs/PROGRESS.md` first.
**Syllabus link:** Unit III (YOLO object detection), Unit IV (transfer learning and fine-tuning).

This phase produces `ml/train_colab.ipynb`. The user opens it in Colab, selects a T4 GPU, and clicks
**Run all**. Nothing else should be required of them.

## What we're doing, in syllabus terms
Start from YOLO26n pretrained on COCO (transfer learning) and fine-tune all layers on FPI-Det so
it detects two classes: `head` and `phone`. COCO already knows "cell phone" but not "head", and it
struggles with small or partly hidden phones. Fine-tuning fixes both.

## Use Context7 first
Look up current Ultralytics docs for YOLO26 model names, `model.train()` arguments, and resuming
training. Don't copy YOLOv8-era arguments blindly.

## Notebook requirements
Clearly titled cells, in this order. Each cell starts with a one-line markdown note explaining what it
does in plain words, so the notebook doubles as lab documentation.

1. **Config (the only cell the user may edit):** `REPO_URL`,
   `DRIVE_DIR = "/content/drive/MyDrive/focus-coach"`, `EPOCHS = 60`, `IMGSZ = 640`,
   `MODEL = "yolo26n.pt"`, `PATIENCE = 15`, `SEED = 42`.
2. **GPU check:** stop with a clear message if no GPU is attached.
3. **Mount Drive** and create `DRIVE_DIR`.
4. **Get code:** clone the user's repo. If it's private, read a GitHub token from Colab Secrets
   (`google.colab.userdata.get("GITHUB_TOKEN")`) and explain in markdown how to add it.
5. **Get data:**
   - If `DRIVE_DIR/fpi_yolo.zip` exists, unzip it to `/content/data/fpi_yolo` and skip the rest.
   - Else: clone FPI-Det, download the images with `gdown --fuzzy`, run `python ml/prepare_dataset.py`
     with Colab paths, then zip the result to `DRIVE_DIR/fpi_yolo.zip` for later sessions.
6. **Install** the pinned `requirements.txt`.
7. **Look at the data:** show 8 training images with their boxes drawn (sanity check before spending GPU time).
8. **Train:**
   - `project=f"{DRIVE_DIR}/runs"`, `name="yolo26n_fpi"`, `exist_ok=True`, `seed=SEED`,
     `deterministic=True`, `patience=PATIENCE`, `imgsz=IMGSZ`, `epochs=EPOCHS`, batch via AutoBatch.
   - Keep Ultralytics' default augmentations. Record them in `train_summary.json` (the report
     discusses augmentation briefly).
   - **Resume:** if `runs/yolo26n_fpi/weights/last.pt` exists on Drive, resume from it. Free Colab
     sessions disconnect, so this is mandatory.
9. **Validate** `best.pt` on the **val** split. Print per-class precision, recall, mAP50, and mAP50-95
   in a small table.
10. **Before vs after:** run the COCO-pretrained `yolo26n.pt` and the fine-tuned `best.pt` on the same
    6 val images and show them side by side. This becomes a report figure.
11. **Save to Drive:** `best.pt`, `results.csv`, Ultralytics training plots, the before/after figure, and
    `train_summary.json` (final metrics, epochs run, train time, GPU name, ultralytics version,
    augmentation settings).
12. **Final markdown cell:** exact steps to download `best.pt` into the local repo at `models/best.pt`,
    and the JSON + plots + figure into `results/train/`.

## Also write
`docs/COLAB.md`: a 6–10 line guide (open the notebook from GitHub in Colab → Runtime → T4 GPU → add the
secret if the repo is private → Run all → what to download → if the session dies, just Run all again
and it resumes).

## Done when
- [ ] Notebook runs top to bottom on a fresh Colab runtime with only the config cell edited
- [ ] A disconnected session resumes from `last.pt` on Run all
- [ ] `fpi_yolo.zip` is cached on Drive after the first run
- [ ] Final cells print the per-class val table and save the before/after figure
- [ ] `docs/COLAB.md` written; PROGRESS.md updated

When the notebook is ready, **stop** and tell the user to run it. Resume with Phase 3 once
`models/best.pt` exists locally.
