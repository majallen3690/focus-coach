# Phase 1 — Dataset prep

## Goal
Turn the FPI-Det download into a clean YOLO-format dataset at `data/fpi_yolo/` with a `fpi.yaml`,
plus held-out splits for the usage-classification evaluation, and write dataset stats.

## What is known (inspected from the real files, see PROGRESS.md)
- The download (`data/raw/reorganized_dataset/{train,val,test}/{images,labels}`) is already in YOLO
  format: 18,800 / 1,730 / 2,349 images. Class `0` = phone, `1` = head, same ids as the COCO JSON.
- `classification_task/label.csv` has 4,079 rows and is **exactly the val + test images**, so nothing
  in train is used for usage evaluation. **`class_id` 1 = using phone** (not 0, despite a script
  docstring in the repo): ~97% of those images have a phone box.
- Label files have defects: several boxes on one line, ~40 boxes slightly outside the image, 3
  degenerate boxes.
- Images are person crops from surveillance footage, not laptop-webcam frames.

## Splits (decided from the above, no re-splitting)
- `det_train` = train, `det_val` = val, `det_test` = test.
- `usage_val` = val rows of `label.csv`, `usage_test` = test rows. Val is used for detector
  checkpoint selection and rule tuning; test only for final numbers.

## Steps
1. Inspect `data/raw/` and record layout, counts and defects in PROGRESS.md. (done)
2. `ml/prepare_dataset.py`:
   - parse labels (split joined lines), clip boxes to the image, drop degenerate boxes, log counts;
   - write `data/fpi_yolo/{images,labels}/{train,val,test}/` and `fpi.yaml` (no `path:` key, so the
     folder works unchanged on Colab);
   - write `data/fpi_yolo/usage_labels.csv` (`image_name,split,using_phone`) with the polarity
     normalised, and fail if the polarity check disagrees with the boxes;
   - write `results/dataset_stats.json`, `results/splits.json`, and one contact sheet per split in
     `results/label_check/` for checking boxes by eye;
   - zip the dataset to `data/fpi_yolo.zip` for Colab.

## Outputs
`ml/prepare_dataset.py`, `data/fpi_yolo/` + `fpi.yaml` + `usage_labels.csv`, `data/fpi_yolo.zip`,
`results/dataset_stats.json`, `results/splits.json`, `results/label_check/`.

## Done when
- [x] Raw data layout and label formats recorded in PROGRESS.md
- [x] Leakage between `label.csv` and detection training checked (none: it is val + test only)
- [x] `python ml/prepare_dataset.py` runs from the repo root and is idempotent (run twice)
- [x] Every image in each split has a label file (empty file allowed) and counts match the stats
- [x] 10 random images per split drawn with their boxes and checked by eye (`results/label_check/`)
- [x] `fpi.yaml` loads with Ultralytics (`ultralytics.data.utils.check_det_dataset`)
- [x] PROGRESS.md entry added
