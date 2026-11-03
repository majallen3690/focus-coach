"""The Gradio demo's handlers, called directly (no browser, no network). Needs gradio and models/best.pt."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

gr = pytest.importorskip("gradio")
from app.pipeline import FramePipeline, WEIGHTS  # noqa: E402
from demo import gradio_app  # noqa: E402
from ml import rules  # noqa: E402

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/best.pt is not available")
ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "demo/examples/images/03_on_phone.jpg"


@pytest.fixture(scope="module")
def setup():
    pipeline = gradio_app.SharedPipeline(FramePipeline(rules.load_config(), weights=WEIGHTS))
    return pipeline.cfg, pipeline  # the pipeline's config carries the app's phone confidence, as in build_demo


def test_live_session_start_frames_end_writes_a_report(setup, monkeypatch):
    cfg, pipeline = setup
    monkeypatch.setattr(gradio_app, "MIN_FRAME_GAP_MS", 0)  # frames here come back to back; none should be skipped
    live_frame, live_end = gradio_app.make_live_handlers(cfg, pipeline)
    blank = np.zeros((120, 160, 3), np.uint8)
    coach = None
    for _ in range(5):  # the first frame starts the session
        shown, status, counters, fps, coach = live_frame(blank, coach, False)
    assert shown.shape == (120, 160, 3) and "FOCUSED" in status and "fc-s-focused" in status
    assert "Pickups" in counters and "<b>0</b>" in counters
    coach, report, events, message = live_end(coach)
    assert Path(report).exists() and Path(events).exists() and "0 pickups" in message
    assert "No session is running" in live_end(coach)[3]  # ending twice is harmless


def test_frames_that_queued_up_are_skipped(setup):
    cfg, pipeline = setup
    live_frame, _ = gradio_app.make_live_handlers(cfg, pipeline)
    blank = np.zeros((120, 160, 3), np.uint8)
    coach = live_frame(blank, None, False)[-1]
    assert isinstance(live_frame(blank, coach, False)[0], type(gr.skip())) and coach.frames == 1


def test_live_frame_without_input_does_nothing(setup):
    cfg, pipeline = setup
    live_frame, _ = gradio_app.make_live_handlers(cfg, pipeline)
    assert live_frame(None, None, False)[0] is None


def test_photo_tab_gives_a_verdict_for_the_example(setup):
    cfg, pipeline = setup
    classify = gradio_app.make_image_handler(cfg, pipeline)
    image, verdict = classify(cv2.cvtColor(cv2.imread(str(EXAMPLE)), cv2.COLOR_BGR2RGB))
    assert image.ndim == 3 and verdict.startswith("USING a phone") and "ON_PHONE" in verdict


def test_footer_headline_comes_from_the_results_file():
    accuracy = json.loads((ROOT / "results/eval/classification.json").read_text())["methods"]["B-rules"]["metrics"]["accuracy"]
    assert f"{accuracy:.3f}" in gradio_app.footer_text(WEIGHTS)


def test_without_ffmpeg_the_video_is_left_as_it_is(monkeypatch, tmp_path):
    monkeypatch.setattr(gradio_app.shutil, "which", lambda name: None)
    assert gradio_app.to_browser_mp4(tmp_path / "a.mp4", tmp_path / "b.mp4") is False


def test_the_page_builds():
    assert isinstance(gradio_app.build_demo(), gr.Blocks)


def test_launch_options_carry_the_look_of_the_page():
    options = gradio_app.LAUNCH_OPTIONS
    assert options["theme"] is gradio_app.THEME and ".fc-hero" in options["css"] and ".fc-s-call" in options["css"]


def test_verdict_card_is_red_for_a_phone_and_green_otherwise():
    using = gradio_app.verdict_html("USING a phone | people: ON_CALL")
    clear = gradio_app.verdict_html("NOT using a phone | people: FOCUSED")
    assert "fc-v-using" in using and "ON CALL" in using and "fc-v-clear" in clear


def test_photo_tab_finds_a_dark_phone_held_at_the_ear(setup):
    cfg, pipeline = setup
    classify = gradio_app.make_image_handler(cfg, pipeline)
    photo = ROOT / "demo/examples/images/05_on_call_side_profile.jpg"
    _, verdict = classify(cv2.cvtColor(cv2.imread(str(photo)), cv2.COLOR_BGR2RGB))
    assert verdict.startswith("USING a phone") and "ON_CALL" in verdict


def test_a_video_file_can_be_analysed_without_a_browser(setup):
    cfg, pipeline = setup
    video, report, files, message = gradio_app.analyse_video(ROOT / "demo/examples/scripted_clip.mp4", cfg, pipeline)
    assert video.exists() and report.exists() and len(files) == 3 and "pickups" in message
