# Phase 7 — Focus Coach Demo on Colab (Gradio, Optional Add-on)

Read `CLAUDE.md` and `docs/PROGRESS.md` first. **Start only after Phase 6 is done.** The graded project is
the OpenCV desktop app. This phase lets anyone run Focus Coach from a browser with no install: open the
notebook in Colab, click **Run all**, and open the link it prints.

**Rule for this phase:** the demo is a *new front end*, not new logic. It reuses `models/best.pt`,
`shared/rules.json`, `ml/rules.py`, `ml/smoothing.py`, `ml/enhance.py`, and `app/report.py`. If any of them
need changes to work here, only make changes that separate logic from display (e.g. `report.py` returning an
image instead of opening a window), keep behavior identical, and keep all existing tests passing.

## Use Context7 first
Check the current Gradio docs for: webcam streaming (`gr.Image(sources=["webcam"], streaming=True)` and the
`.stream()` event with `stream_every` / `time_limit`), `gr.State`, `gr.Video`, `gr.File`, `gr.Progress`,
`demo.launch(share=True)` in Colab, and queue settings. The streaming API changed across Gradio versions.
Pin the version you verify.

## Part A — `app/offline.py` (shared video processing, no UI)
`process_video(path, sample_fps=10, cfg) -> (annotated_frames_iter, events, summary)`:
- Reads the video with OpenCV and samples frames at `sample_fps`, using **video timestamps** (frame index / fps)
  so results are deterministic and don't depend on processing speed.
- Per frame: brightness check → `enhance` (per `rules.json → app`) → model → rules (`"largest"` head) → smoother → session.
- Returns the same `events` and `summary` structures the desktop app writes.

CLI: `python -m app.offline path/to/clip.mp4 --out sessions/offline_<name>/` writes `events.csv`,
`summary.json`, `report.png`, and `annotated.mp4`. Add a pytest that runs it on a 3-second synthetic or
fixture clip and checks the output files exist and parse.

## Part B — `demo/gradio_app.py`
`build_demo(model_path, rules_path) -> gr.Blocks`, with three tabs.

**1. Live (webcam)**
- Webcam stream from the user's browser → server, about 4–6 frames/s (`stream_every` ≈ 0.2 s).
- Per-user state in `gr.State`: smoother + session. Timestamps come from server receive time.
- Outputs: annotated frame (boxes + status banner drawn with the same `app/hud.py` functions), a status line
  (FOCUSED / ON PHONE / ON CALL / AWAY), counters (elapsed, pickups, phone time, current streak), and the
  measured frames/s.
- Buttons: **Start**, **End session**. End → the report image from `app/report.py` + `events.csv` download.
- Checkbox: "Force low-light enhancement" (default follows `rules.json`).
- Visible note under the video: "Frames are sent to this Colab session for processing and are not stored."

**2. Video (upload) — the reliable demo mode**
- Upload an mp4 → `app/offline.process_video` with a `gr.Progress` bar.
- Outputs: annotated video, report image, `events.csv` + `summary.json` downloads.
- Annotated video must play in the browser: write frames with OpenCV, then convert to H.264 with `ffmpeg`
  (`-c:v libx264 -pix_fmt yuv420p -movflags +faststart`). Colab has ffmpeg.
- If `demo/examples/demo.mp4` exists (the clip from Phase 5's `make_demo_clip.md`), list it as a clickable example.

**3. Image (upload)**
- Upload a photo → boxes, each person's state, and the image-level "using / not using" verdict. No smoothing.
- Include 4 example images from the FPI-Det held-out split (one per state + one tricky case).

Layout: a short header ("Focus Coach — Mobile Phone Usage Detection"), one-line instructions per tab, and a
footer with the model name, device (GPU/CPU), and the project's headline accuracy pulled from
`results/eval/classification.json`. **Never typed by hand.**

`python -m demo.gradio_app` must also run locally (no share link) for testing.

## Part C — `demo/focus_coach_colab.ipynb`
Clearly titled cells, each with a one-line markdown explanation:
1. **Config (the only cell the user may edit):** `REPO_URL`,
   `MODEL_PATH = "/content/drive/MyDrive/focus-coach/runs/yolo26n_fpi/weights/best.pt"`, `USE_DRIVE = True`.
2. **Runtime check:** print GPU name, or a warning that CPU works but is slower.
3. **Get code:** clone the repo (GitHub token from Colab Secrets if private, same as Phase 2).
4. **Get model:** from Drive if `USE_DRIVE`, otherwise show a file-upload widget for `best.pt`.
5. **Install** the pinned requirements + the pinned Gradio version.
6. **Smoke test:** run the model on one example image and print detections. Fail early with a clear
   message if anything's wrong.
7. **Launch:** `build_demo(...).queue().launch(share=True, debug=True)`. Print the public link prominently,
   with a note that the link is temporary and stops working when the Colab session ends.

## Part D — Docs
- `docs/COLAB_DEMO.md`: a 6–10 line guide (open notebook → Runtime → T4 GPU → Run all → click the link → allow
  camera → use the Video tab if the live feed is laggy).
- README: a "Run it in Colab" section with an "Open in Colab" badge for the notebook.
- `report/viva.md`: add 3 Q&As:
  - Why is the Colab demo slower than the desktop app? (Every frame travels over the internet both ways.)
  - Where does the video go? (To the user's own temporary Colab session, processed in memory, not stored.
    The desktop app keeps everything on the laptop.)
  - Why Gradio over Streamlit? (Built-in webcam streaming and share links in Colab, with no tunnel setup.)

## Done when
- [x] `pytest` passes, including the `app/offline.py` test; all earlier tests still pass (84 tests)
- [ ] (NOT verified: needs the user's Colab and GitHub repo) On a fresh Colab T4 runtime, Run all → the public link opens and all three tabs work
- [ ] (NOT verified in a browser with a real camera; the handlers passed a scripted sequence locally: 1 pickup of 5.0 s, glance ignored, about 30 frames/s server side) Live tab runs at ≥ 3 frames/s; a ~5 s phone pickup logs one event; a 0.5 s glance logs none
- [x] Video tab returns a browser-playable annotated video, a report, and downloads (checked through the Gradio API on a local run: H.264 yuv420p video, report.png, events.csv, summary.json; not played in a browser)
- [x] **Consistency check:** the same clip through the Video tab and through `python -m app.offline` locally gives
      the same pickup count, with event start times within 0.5 s
- [x] `docs/COLAB_DEMO.md`, the README section, and the viva Q&As written; PROGRESS.md updated
