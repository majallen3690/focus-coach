# Phase 5 — Focus Coach Desktop App (OpenCV)

Read `CLAUDE.md` and `docs/PROGRESS.md` first. Requires `models/best.pt` and a `shared/rules.json`
with the `app` section from Phase 4.
**Syllabus link:** Unit I (image acquisition, CV pipeline, OpenCV), Unit II (live enhancement),
Unit III (YOLO inference), Unit V (activity recognition over time; real-world application).

## What the user experiences
1. Run `python -m app.focus_coach`. A window opens with the webcam feed.
2. Press **S** to start a session. A status label shows FOCUSED / ON PHONE / ON CALL / AWAY.
3. Pick up the phone for a few seconds: the label turns amber and the pickup counter goes up.
   A half-second glance does nothing.
4. Press **E** to end. A report image opens and is saved with the session data.

## Use Context7 first
Check current docs for `cv2.VideoCapture` on Windows (DirectShow backend), Ultralytics `predict` on
CPU (arguments, result objects), and matplotlib for the report.

## Modules
**`app/capture.py`**: webcam/video reader on its own thread that keeps only the **latest** frame
(drop old frames; never build up lag). Windows: open the webcam with `cv2.CAP_DSHOW`. Supports
`--source 0` (webcam) or `--source path/to/video.mp4`. Clear error if the camera can't be opened.

**`ml/smoothing.py`** (pure Python, testable with fake timestamps)
- Sliding window of the last `smoothing.window_ms` of per-frame states, timestamped.
- The displayed state becomes X only when X makes up ≥ `smoothing.enter_ratio` of the window.
- No head detected for longer than `smoothing.away_ms` → `AWAY` (shown, but not counted as phone time).
  Add `"away_ms": 3000` to the `smoothing` section of `shared/rules.json` if it's missing.
- Emits `distraction_start {type, t}` and `distraction_end {t, duration_ms}`.
- `ml/tests/test_smoothing.py`: a glance shorter than the window → no event; sustained use → exactly
  one event with the correct duration; flicker at the boundary → no double count; AWAY handling.

**`app/session.py`**: session state: start/end time, current state, event list, counters (pickups,
phone time, longest focus streak, current streak), and pause/resume.

**`app/hud.py`**: everything drawn with OpenCV:
- head and phone boxes with class + confidence (debug mode only, toggled with **D**)
- a large status banner (green FOCUSED / amber ON PHONE / red ON CALL / grey AWAY)
- counters: elapsed time, pickups, phone time, current focus streak
- footer: FPS, inference ms, and enhancement status ("Low light: CLAHE on" when active)
- before a session starts, an on-screen tip: "Make sure the camera sees your face and the desk in front of you."

**`app/report.py`**: at session end, writes to `sessions/<YYYY-MM-DD_HH-MM-SS>/`:
- `events.csv` (start, end, type, duration_s)
- `summary.json` (total, focused, phone time, % focused, pickups, longest streak, avg distraction length)
- `report.png` (matplotlib): a stats panel plus a timeline with colored bands for FOCUSED / ON_PHONE /
  ON_CALL / AWAY across the session
Then opens `report.png` in an OpenCV window.

**`app/focus_coach.py`**: main loop:
1. Grab the latest frame.
2. If `to_gray_mean(frame) < app.low_light_threshold` and `app.enhance_method != "none"`, apply
   `enhance()` from `ml/enhance.py`. **C** toggles this manually.
3. Run `best.pt` on CPU (imgsz from config; default 640, allow 480 for speed).
4. `rules.filter_dets` → `pick_user_head("largest")` → `classify_person`.
5. Feed the per-frame state to the smoother; update the session; draw the HUD.
Keys: **S** start/pause, **E** end session, **D** debug boxes, **C** toggle enhancement, **Q** quit
(prompt to save if a session is running).
Flags: `--source`, `--imgsz`, `--save-video path.mp4` (writes the annotated feed; off by default).

## Privacy
Frames are processed in memory and never saved unless `--save-video` is passed. Only events and
summary numbers are written to `sessions/`. Show this in the start screen tip and the README.

## Backup demo
Write `app/make_demo_clip.md`: a 1-paragraph guide for the user to record one 60-second clip of
themselves (focused → glance → phone pickup → call → focused) so the viva demo can run from
`--source demo.mp4` if the webcam misbehaves.

## Done when
- [x] `pytest ml/tests` passes, including `test_smoothing.py`
- [ ] App runs at ≥ 8 FPS on the user's laptop CPU (record the actual FPS and CPU in `results/eval/summary.md`; drop to imgsz 480 if needed and record that)
- [x] (skipped at the user's request, not verified live) 5-minute self-test by the user: a ~5 s phone pickup logs exactly one event; a 0.5 s glance logs none; covering the camera triggers AWAY
- [ ] Ending a session writes `events.csv`, `summary.json`, and `report.png`, and opens the report
- [x] Works with `--source video.mp4` as well as the webcam (video verified; the live window is part of the self-test)
- [ ] PROGRESS.md updated

When the code is ready, **stop** and ask the user to run the self-test. Fix any issues they report
before marking the phase done.
