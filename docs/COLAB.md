# Training on Colab: quick guide

1. Push this repo to GitHub. In Colab: File → Open notebook → GitHub tab → pick `ml/train_colab.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. In the first code cell, set `REPO_URL` to your repo. Leave the other settings unless you know why to change them.
4. If the repo is private: left bar → key icon → Add new secret → name `GITHUB_TOKEN`, value a GitHub token with read access, *Notebook access* on.
5. Runtime → **Run all**. Allow Drive access when asked. The first run builds the dataset and saves `fpi_yolo.zip` on Drive; later runs reuse it.
   To skip that download, put an existing `fpi_yolo.zip` (made by `python ml/prepare_dataset.py`) in the `DRIVE_DIR` folder before you start.
6. When it finishes, open `focus-coach/outputs/` on Drive. Save `best.pt` as `models/best.pt` in the repo and everything else in `results/train/`.
7. If the session dies or disconnects, just **Run all** again. It resumes from the last checkpoint on Drive, and skips training if the run had already finished.
8. If Colab asks to restart the runtime after the install cell, restart and Run all again. Nothing is lost.
