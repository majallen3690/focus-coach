# Phase 2b — Catch-up (training already done with the earlier Phase 2 prompt)

Read `CLAUDE.md` and `docs/PROGRESS.md` first. The user already trained YOLO26n with an earlier version
of `docs/02-train-colab.md`. The training settings were the same, so **do not retrain**. This phase only
generates the outputs the current Phase 2 expects but the earlier run didn't produce.

## Step 1 — Check what exists (stop and ask the user for anything missing)
- `models/best.pt` (required)
- From the Colab run folder on Drive (`MyDrive/focus-coach/runs/yolo26n_fpi/`), copied into `results/train/`:
  `results.csv`, the Ultralytics training plots, `args.yaml`, and `train_summary.json` if it exists
- `data/fpi_yolo/` with the val split available locally

If `args.yaml` or the plots are missing, give the user the exact Drive path to download them from.

## Step 2 — `ml/phase2_catchup.py` (CPU, one command)
1. **Per-class validation table:** run `best.pt` on the **val** split and save per-class precision,
   recall, mAP50, and mAP50-95 for `head` and `phone` to `results/train/val_per_class.json`.
   Print it as a table.
2. **Before/after figure:** run COCO-pretrained `yolo26n.pt` and `best.pt` on the same 6 val images
   (seed 42; prefer images with small or partly hidden phones) and save them side by side to
   `results/train/before_after.png`.
3. **Training summary:** build or complete `results/train/train_summary.json` with: final val metrics,
   epochs actually run (from `results.csv`), the augmentation settings (from `args.yaml`), and the
   ultralytics version. Mark any field that can't be recovered (e.g. GPU name, train time) as `null`
   with a note. **Don't guess.**

## Step 3 — Keep the repo consistent
Update `ml/train_colab.ipynb` to match the current `docs/02-train-colab.md` (add the data-preview,
per-class table, and before/after cells) so anyone re-running it gets the same outputs. **Don't run it.**
Add a note at the top of the notebook that the reported model was trained before these cells were added.

## Done when
- [x] `results/train/` contains `val_per_class.json`, `before_after.png`, `train_summary.json`, `results.csv`, and the training plots
- [x] Notebook updated to match the current Phase 2 spec
- [x] PROGRESS.md notes that training used the earlier notebook and these outputs were regenerated locally

Then continue with `docs/03-evaluate.md`.
