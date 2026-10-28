# Run the browser demo on Colab

1. Push this repo to GitHub, with `demo/` and `results/eval/classification.json` included. Keep `models/best.pt` out of it.
2. In Colab: File > Open notebook > GitHub tab > pick `demo/focus_coach_colab.ipynb`.
3. Runtime > Change runtime type > **T4 GPU** (a CPU works but is slower).
4. In the config cell, set `REPO_URL`. Leave `MODEL_PATH` as it is if Phase 2 saved the model on your Drive; otherwise set `USE_DRIVE = False` and upload `best.pt` when asked.
5. Runtime > **Run all**. Allow Drive access if asked. If Colab asks to restart the runtime after the install cell, restart and Run all again.
6. Click the public link printed at the end. It is temporary and stops working when the Colab session ends.
7. On the **Live** tab allow the camera, press **Start**, pick up your phone, press **End session** for the report.
8. If the live feed is laggy (every frame travels over the internet both ways), use the **Video** tab with a recorded clip; it is the reliable mode. The **Image** tab judges a single photo.
9. Locally, without a link: `pip install -r requirements-demo.txt` then `python -m demo.gradio_app`, and open http://127.0.0.1:7860.
