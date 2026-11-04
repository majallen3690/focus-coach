"""Focus Coach in the browser (Gradio): a new front end for the same model, rules, smoothing and report code.

Run locally (no public link) from the repo root:
    python -m demo.gradio_app                 --port 7861 to change the port; and open http://127.0.0.1:7860
In Colab, use demo/focus_coach_colab.ipynb. Three tabs:
    Live     webcam frames travel from the browser to this computer and back (about 5 per second)
    Video    upload a clip; it is processed with app/offline.py (the same code as `python -m app.offline`)
    Image    upload one photo; shows the boxes, each person's state and the image-level verdict, with no smoothing
Frames are processed in memory. Uploaded files and reports live in a temporary folder of this session.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import hud  # noqa: E402
from app.capture import Frame  # noqa: E402
from app.focus_coach import FocusCoach  # noqa: E402
from app.offline import run_to_folder  # noqa: E402
from app.pipeline import FramePipeline, FrameResult, WEIGHTS  # noqa: E402
from app.report import write_report  # noqa: E402
from ml import rules  # noqa: E402

RULES_PATH = ROOT / "shared/rules.json"
CLASSIFICATION = ROOT / "results/eval/classification.json"
EXAMPLES = ROOT / "demo/examples"
STREAM_EVERY_S = 0.4        # about 2.5 frames per second from the browser: more than a tunnel and a CPU can keep up with only builds a backlog
STREAM_TIME_LIMIT_S = 900   # one stream is cut after this long; it restarts by itself. No concurrency limit, so a second open tab cannot block the stream
PRIVACY_NOTE = "Frames are processed in memory and are not stored."
LIVE_START_HINT = "Click Record on the webcam to start"
LIVE_KEYS_HINT = "End session: button below"
WEBCAM_SIZE = {"width": 480, "height": 360}  # small frames travel faster to and from the server
MIN_FRAME_GAP_MS = 300      # frames that queued up behind a slow one are skipped, so the picture never falls behind
VIDEO_MAX_SIDE = 640        # the annotated result video is shrunk to this, so it downloads fast even through a slow tunnel
VIDEO_CRF = 30              # H.264 quality: higher means a smaller file

# The look of the page. Gradio 6 takes theme and css at launch(), so every launch call passes LAUNCH_OPTIONS.
THEME = gr.themes.Soft(primary_hue="teal", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
CSS = (Path(__file__).with_name("style.css")).read_text(encoding="utf-8")
LAUNCH_OPTIONS = {"theme": THEME, "css": CSS, "footer_links": []}
STATE_CLASSES = {"Not started": "idle", "PAUSED": "paused", "FOCUSED": "focused", "ON PHONE": "phone", "ON CALL": "call", "AWAY": "away"}


class SharedPipeline:
    """One loaded model for all users; only one frame is in the detector at a time."""

    def __init__(self, pipeline: FramePipeline) -> None:
        self._pipeline, self._lock = pipeline, threading.Lock()
        self.imgsz, self.cfg = pipeline.imgsz, pipeline.cfg

    def process(self, frame: np.ndarray, enhance_mode: str = "auto") -> FrameResult:
        with self._lock:
            return self._pipeline.process(frame, enhance_mode)


def to_browser_mp4(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 so every browser can play it. Returns False when ffmpeg is not available."""
    if shutil.which("ffmpeg") is None:
        return False
    # fast preset and a higher CRF: a small file matters more than fine detail, because it has to travel through a tunnel
    run = subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", str(VIDEO_CRF),
                          "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)], capture_output=True, text=True)
    return run.returncode == 0 and dst.exists()


def device_name() -> str:
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (works, but is slower than a GPU)"


def headline_metrics() -> dict | None:
    """The project's headline classification numbers, read from results/ (never typed here)."""
    if not CLASSIFICATION.exists():
        return None
    data = json.loads(CLASSIFICATION.read_text())
    return {"images": data["images"], **data["methods"]["B-rules"]["metrics"]}


def footer_text(model_path: Path) -> str:
    """Model, device and the project's headline accuracy."""
    text = f"Model: {model_path.name} (YOLO26n fine-tuned on FPI-Det) | Device: {device_name()}"
    m = headline_metrics()
    if m:
        text += f" | Detection + rules on {m['images']:,} held-out photos: accuracy {m['accuracy']:.3f}, F1 {m['f1']:.3f}"
    return text


# --- small HTML pieces ----------------------------------------------------------------------------------------

def hero_html() -> str:
    pills = ""
    m = headline_metrics()
    if m:
        pills = (f'<div class="fc-pill"><b>{m["accuracy"]:.1%}</b><span>accuracy</span></div>'
                 f'<div class="fc-pill"><b>{m["f1"]:.3f}</b><span>F1 score</span></div>'
                 f'<div class="fc-pill"><b>{m["images"]:,}</b><span>held-out photos</span></div>')
    return ('<div class="fc-hero"><div>'
            '<div class="fc-eyebrow">Computer Vision mini project</div>'
            '<h1>Focus Coach</h1>'
            '<p>A webcam watches you study, spots when you pick up your phone, and tells you how much time it cost.</p>'
            f'</div><div class="fc-pills">{pills}</div></div>')


def steps_html(steps: list[tuple[str, str]]) -> str:
    items = "".join(f"<li><span>{i}</span><div><b>{title}</b><small>{hint}</small></div></li>" for i, (title, hint) in enumerate(steps, 1))
    return f'<ol class="fc-steps">{items}</ol>'


def card_title(text: str) -> str:
    return f'<div class="fc-card-title">{html.escape(text)}</div>'


def callout(text: str, kind: str = "info") -> str:
    return f'<div class="fc-callout fc-{kind}">{html.escape(text)}</div>'


def verdict_html(text: str) -> str:
    """The photo verdict as a big coloured card: red when a phone is in use, green when not."""
    headline, _, detail = text.partition(" | ")
    kind = "using" if headline.startswith("USING") else "clear"
    detail_html = f'<div class="fc-verdict-detail">{html.escape(detail.replace("_", " "))}</div>' if detail else ""
    return f'<div class="fc-verdict fc-v-{kind}"><div class="fc-verdict-title">{html.escape(headline)}</div>{detail_html}</div>'


def legend_html() -> str:
    rows = [("var(--fc-focused)", "Focused", "No phone in use. A phone lying far away on the desk does not count."),
            ("var(--fc-phone)", "On phone", "A phone is held in front of you, below your head."),
            ("var(--fc-call)", "On call", "A phone is held at your ear or face."),
            ("var(--fc-away)", "Away", "No head seen for more than 3 seconds.")]
    items = "".join(f'<div><span class="fc-chip" style="background:{color}"></span><span><b>{name}</b><small>{what}</small></span></div>'
                    for color, name, what in rows)
    return (f'<div class="fc-legend">{items}</div>'
            '<p class="fc-pipeline">How it decides: the picture is brightened if it is dark (CLAHE), a YOLO26n detector finds heads and phones, '
            'simple rules turn the boxes into a state, and a state only counts after it holds for about 1.5 seconds, '
            'so a quick glance is ignored.</p>')


# --- tab 1: live ----------------------------------------------------------------------------------------------

def new_coach(cfg: dict, pipeline: SharedPipeline) -> FocusCoach:
    coach = FocusCoach(cfg, pipeline=pipeline)
    coach.debug = True
    coach.start_hint, coach.keys_hint = LIVE_START_HINT, LIVE_KEYS_HINT
    return coach


def status_line(coach: FocusCoach | None) -> str:
    if coach is None or not coach.session.running:
        return "Not started"
    return "PAUSED" if coach.session.paused else hud.STATE_LABELS[coach.session.state]


def status_html(coach: FocusCoach | None) -> str:
    label = status_line(coach)
    return f'<div class="fc-state fc-s-{STATE_CLASSES[label]}"><span class="fc-dot"></span>{label}</div>'


def counters_html(coach: FocusCoach | None) -> str:
    if coach is None or not coach.session.running:
        values = ["00:00", "0", "00:00", "00:00"]
    else:
        s = coach.session
        values = [hud.clock_text(s.elapsed_ms), str(s.pickups), hud.clock_text(s.phone_ms), hud.clock_text(s.current_streak_ms)]
    labels = ["Time", "Pickups", "Phone time", "Focus streak"]
    tiles = "".join(f'<div class="fc-tile"><span>{label}</span><b>{value}</b></div>' for label, value in zip(labels, values))
    return f'<div class="fc-tiles">{tiles}</div>'


def speed_html(text: str) -> str:
    return f'<div class="fc-speed">{text}</div>'


def make_live_handlers(cfg: dict, pipeline: SharedPipeline):
    def live_frame(image: np.ndarray | None, coach: FocusCoach | None, force_enhance: bool):
        if image is None:
            return None, status_html(coach), counters_html(coach), speed_html(""), coach
        coach = coach or new_coach(cfg, pipeline)
        coach.enhance_mode = "on" if force_enhance else "auto"
        now_ms = time.monotonic() * 1000  # the server's clock: when the frame arrived
        if coach.frames and now_ms - coach.last_t_ms < MIN_FRAME_GAP_MS:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), coach
        if not coach.session.running:  # Record starts the session; no separate Start button
            coach.session.start(now_ms)
        out = coach.handle(Frame(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), now_ms, coach.frames))
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB), status_html(coach), counters_html(coach), speed_html(f"{coach.fps:.1f} frames/s"), coach

    def live_end(coach: FocusCoach | None):
        if coach is None or not coach.session.running:
            return coach, None, None, callout("No session is running. Click Record on the webcam first.", "warn")
        coach.session.end(coach.last_t_ms)
        folder = write_report(coach.session, {"source": "browser webcam", "imgsz": pipeline.imgsz, **coach.stats(), "settings": cfg},
                              root=Path(tempfile.mkdtemp(prefix="focus_coach_")))
        s = coach.session.summary()
        message = (f"Session ended: {s['pickups']} pickups, {s['phone_s']} s on phone, {s['away_s']} s away, "
                   f"{s['percent_focused']}% focused, "
                   f"longest focus {s['longest_focus_streak_s']} s. While Record is on, a new session starts by itself.")
        coach.session = type(coach.session)(cfg["smoothing"])  # a fresh session; the next frame starts it
        return coach, str(folder / "report.png"), str(folder / "events.csv"), callout(message)

    return live_frame, live_end


# --- tab 2: video ---------------------------------------------------------------------------------------------

def analyse_video(path: str | Path, cfg: dict, pipeline: SharedPipeline, on_progress=None) -> tuple[Path, Path, list[str], str]:
    """Process one video file; returns (annotated video, report picture, download files, one-line summary). No browser needed."""
    out = Path(tempfile.mkdtemp(prefix="focus_coach_video_"))
    result = run_to_folder(Path(path), out, coach=new_coach(cfg, pipeline), on_progress=on_progress, max_side=VIDEO_MAX_SIDE)
    playable = out / "annotated_h264.mp4"
    video = playable if to_browser_mp4(out / "annotated.mp4", playable) else out / "annotated.mp4"
    s = result.session.summary()
    message = (f"{s['pickups']} pickups, {s['phone_s']} s on phone, {s['percent_focused']}% focused, "
               f"longest focus {s['longest_focus_streak_s']} s, {s['away_s']} s away. Video: {video.stat().st_size / 1e6:.1f} MB.")
    if video.name == "annotated.mp4":
        message += " (ffmpeg was not found, so the video may not play in every browser; use the download.)"
    return video, out / "report.png", [str(out / "events.csv"), str(out / "summary.json"), str(video)], message


def make_video_handler(cfg: dict, pipeline: SharedPipeline):
    def process_upload(path: str | None, progress=gr.Progress()):
        if not path:
            raise gr.Error("Upload a video first (an .mp4 works best).")
        try:
            video, report, files, message = analyse_video(path, cfg, pipeline, on_progress=lambda p: progress(p, desc="Analysing"))
        except (FileNotFoundError, ValueError) as error:
            raise gr.Error(str(error)) from error
        return str(video), str(report), files, callout(message)

    return process_upload


# --- tab 3: image ---------------------------------------------------------------------------------------------

def make_image_handler(cfg: dict, pipeline: SharedPipeline):
    def classify_photo(image: np.ndarray | None):
        if image is None:
            raise gr.Error("Upload a photo first.")
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = pipeline.process(bgr, "auto")
        verdict = rules.classify_image(result.dets, cfg, mode="all")  # every person in the photo, no smoothing
        canvas = result.image.copy()
        for d in result.dets:
            if d["cls"] == "phone":
                x1, y1, x2, y2 = (int(v) for v in d["box"])
                cv2.rectangle(canvas, (x1, y1), (x2, y2), hud.BOX_COLORS["phone"], 2)
        for person in verdict["persons"]:
            x1, y1, x2, y2 = (int(v) for v in person["head"])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), hud.STATE_COLORS[person["state"]], 3)
            label_y = y1 - 6 if y1 > 24 else y1 + 20  # inside the box when the head touches the top edge
            cv2.putText(canvas, person["state"].replace("_", " "), (x1 + 4, label_y), hud.FONT, 0.55,
                        hud.STATE_COLORS[person["state"]], 2, cv2.LINE_AA)
        states = [p["state"] for p in verdict["persons"]]
        if not states:
            text = "No head found, so the rules cannot decide: NOT USING"
        else:
            text = ("USING a phone" if verdict["using"] else "NOT using a phone") + f" | people: {', '.join(states)}"
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), text

    return classify_photo


# --- the page -------------------------------------------------------------------------------------------------

def build_demo(model_path: Path = WEIGHTS, rules_path: Path = RULES_PATH) -> gr.Blocks:
    cfg = json.loads(Path(rules_path).read_text())
    pipeline = SharedPipeline(FramePipeline(cfg, weights=Path(model_path)))
    cfg = pipeline.cfg  # with the app's phone confidence, so the photo tab applies the same rule as the pipeline
    live_frame, live_end = make_live_handlers(cfg, pipeline)
    process_upload = make_video_handler(cfg, pipeline)
    classify_photo = make_image_handler(cfg, pipeline)

    def photo_view(image: np.ndarray | None):
        shown, text = classify_photo(image)
        return shown, verdict_html(text)

    live_steps = steps_html([("Allow the camera", "Click the webcam box on the left"), ("Press Record", "The session starts by itself"),
                             ("Pick up your phone", "Watch the state change on the right"), ("End session", "Get your report below")])

    with gr.Blocks(title="Focus Coach") as demo:
        gr.HTML(hero_html())
        with gr.Tab("Live webcam"):
            gr.HTML(live_steps)
            coach_state = gr.State(None)
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes="fc-card"):
                    gr.HTML(card_title("Your camera"))
                    webcam = gr.Image(sources=["webcam"], streaming=True, type="numpy", show_label=False,
                                      webcam_options=gr.WebcamOptions(constraints={"video": WEBCAM_SIZE}))
                with gr.Column(elem_classes="fc-card"):
                    gr.HTML(card_title("What Focus Coach sees"))
                    annotated = gr.Image(type="numpy", streaming=True, format="jpeg", show_label=False,
                                         placeholder="Press Record on the camera to start")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2, min_width=240):
                    status = gr.HTML(status_html(None))
                with gr.Column(scale=5):
                    counters = gr.HTML(counters_html(None))
                    fps = gr.HTML(speed_html(""))
            end = gr.Button("End session and get the report", variant="primary", elem_classes="fc-primary")
            note = gr.HTML(callout("Live view is slower through a free tunnel or on a CPU. The Video tab is the smooth way to see the whole pipeline. "
                                   + PRIVACY_NOTE))
            with gr.Accordion("Your session report", open=False) as report_box:
                with gr.Row():
                    report_image = gr.Image(type="filepath", show_label=False)
                    events_file = gr.File(label="events.csv")
            with gr.Accordion("Settings", open=False):
                force = gr.Checkbox(label="Force low-light enhancement (default: automatic)", value=False)
            webcam.stream(live_frame, [webcam, coach_state, force], [annotated, status, counters, fps, coach_state],
                          stream_every=STREAM_EVERY_S, time_limit=STREAM_TIME_LIMIT_S, concurrency_limit=None)
            end.click(live_end, [coach_state], [coach_state, report_image, events_file, note]).then(
                lambda: gr.Accordion(open=True), None, report_box)
        with gr.Tab("Video upload"):
            gr.HTML(steps_html([("Choose a clip", "A short .mp4 of someone studying"), ("Analyse", "The whole clip is processed"),
                                ("Read the results", "Annotated video, report and downloads")]))
            with gr.Row(equal_height=False):
                with gr.Column(elem_classes="fc-card"):
                    gr.HTML(card_title("Your video"))
                    video_in = gr.Video(sources=["upload"], show_label=False)
                    run = gr.Button("Analyse video", variant="primary", elem_classes="fc-primary")
                    clips = [p for p in (EXAMPLES / "demo.mp4", EXAMPLES / "scripted_clip.mp4") if p.exists()]  # your own clip first
                    if clips:
                        gr.Examples(examples=[[str(p)] for p in clips], inputs=[video_in], label="Or try an example clip")
                with gr.Column(elem_classes="fc-card"):
                    gr.HTML(card_title("Result"))
                    video_out = gr.Video(show_label=False)
                    video_message = gr.HTML(callout("Results appear here once the video is analysed."))
            with gr.Accordion("Report and downloads", open=False) as video_report_box:
                with gr.Row():
                    report_out = gr.Image(show_label=False, type="filepath")
                    files_out = gr.File(label="Downloads", file_count="multiple")
            run.click(process_upload, [video_in], [video_out, report_out, files_out, video_message]).then(
                lambda: gr.Accordion(open=True), None, video_report_box)
        with gr.Tab("Photo upload"):
            gr.HTML(steps_html([("Choose a photo", "One photo, one or more people"), ("Analyse", "Every person gets a state"),
                                ("Read the verdict", "Boxes show the head and the phone")]))
            with gr.Row(equal_height=False):
                with gr.Column(elem_classes="fc-card"):
                    gr.HTML(card_title("Your photo"))
                    image_in = gr.Image(type="numpy", show_label=False)
                    image_run = gr.Button("Analyse photo", variant="primary", elem_classes="fc-primary")
                    images = sorted((EXAMPLES / "images").glob("*.jpg"))
                    if images:
                        gr.Examples(examples=[[str(p)] for p in images], inputs=[image_in], label="Or try a photo from the held-out set")
                with gr.Column(elem_classes="fc-card"):
                    gr.HTML(card_title("Result"))
                    image_out = gr.Image(type="numpy", show_label=False)
                    verdict_out = gr.HTML(callout("The verdict appears here. No time smoothing is used for single photos."))
            image_run.click(photo_view, [image_in], [image_out, verdict_out])
        with gr.Accordion("What do the colours mean?", open=False):
            gr.HTML(legend_html())
        gr.HTML(f'<div class="fc-footer">{html.escape(footer_text(Path(model_path)))}</div>')
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)
    build_demo().queue().launch(share=args.share, server_port=args.port, **LAUNCH_OPTIONS)


if __name__ == "__main__":
    main()
