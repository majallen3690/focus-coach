"""The webcam source's 'still picture' detection and the on-screen warning, with a fake camera (no hardware, no model)."""
import time

import numpy as np

from app import capture, focus_coach, hud
from ml import rules

CFG = rules.load_config()


class FakeCamera:
    """Stands in for cv2.VideoCapture: `live` frames carry sensor noise, otherwise every frame is the same placeholder."""
    live = False

    def __init__(self, *args) -> None:
        self.rng = np.random.default_rng(0)
        self.still = np.full((48, 64, 3), 12, np.uint8)

    def isOpened(self) -> bool:
        return True

    def set(self, *args) -> bool:
        return True

    def read(self):
        time.sleep(0.005)
        if self.live:
            return True, self.rng.integers(0, 40, (48, 64, 3), dtype=np.uint8)
        return True, self.still.copy()

    def release(self) -> None:
        pass


def open_fake(monkeypatch, live: bool) -> capture.FrameSource:
    monkeypatch.setattr(capture.cv2, "VideoCapture", type("Cam", (FakeCamera,), {"live": live}))
    return capture.FrameSource(0).start()


def test_identical_frames_count_as_a_frozen_camera(monkeypatch):
    source = open_fake(monkeypatch, live=False)
    try:
        time.sleep(0.4)
        assert source.frozen_s >= 0.3
    finally:
        source.release()


def test_a_live_camera_is_never_frozen(monkeypatch):
    source = open_fake(monkeypatch, live=True)
    try:
        time.sleep(0.4)
        assert source.frozen_s < 0.1
    finally:
        source.release()


def test_a_video_file_is_never_reported_as_frozen(tmp_path):
    assert capture.FrameSource(tmp_path / "clip.mp4").frozen_s == 0.0


def test_check_camera_says_ok_for_live_video_and_problem_for_a_still_picture(monkeypatch, capsys):
    monkeypatch.setattr(focus_coach, "CHECK_CAMERA_S", 0.5)
    monkeypatch.setitem(CFG["app"], "camera_frozen_s", 0.2)
    monkeypatch.setattr(capture.cv2, "VideoCapture", type("Cam", (FakeCamera,), {"live": True}))
    assert focus_coach.check_camera("0", CFG) == 0 and "OK" in capsys.readouterr().out
    monkeypatch.setattr(capture.cv2, "VideoCapture", type("Cam", (FakeCamera,), {"live": False}))
    assert focus_coach.check_camera("0", CFG) == 1 and "PROBLEM" in capsys.readouterr().out


def test_the_warning_is_drawn_on_the_picture():
    image = np.zeros((720, 1280, 3), np.uint8)
    hud.draw_camera_problem(image)
    assert image.any()


def test_default_view_outlines_phones_only():
    image = np.zeros((720, 1280, 3), np.uint8)
    dets = [{"cls": "head", "box": [100, 100, 300, 300], "conf": 0.9}, {"cls": "phone", "box": [500, 400, 600, 520], "conf": 0.8}]
    hud.draw_boxes(image, dets, None, only="phone")
    assert image[100:105, 100:300].sum() == 0 and image[400:404, 500:600].sum() > 0
