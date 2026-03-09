"""
tests/test_pipeline.py

Unit tests for ingestion and pose types.
These run WITHOUT a camera or GPU — uses the test source.

Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import pytest

from data_types import Keypoint, PoseFrame, KP, SKELETON_EDGES
from ingestion import FrameIngestion, CapturedFrame, _TestCapture


# ──────────────────────────────────────────
# Type tests
# ──────────────────────────────────────────

def _make_pose(conf=0.9) -> PoseFrame:
    """Create a dummy PoseFrame with all keypoints visible."""
    kps = [Keypoint(x=float(i * 10), y=float(i * 5), confidence=conf) for i in range(17)]
    return PoseFrame(
        timestamp=time.monotonic(),
        wall_time=time.time(),
        frame_index=0,
        keypoints=kps,
        detection_confidence=0.9,
        frame_width=1280,
        frame_height=720,
    )


def test_keypoint_validity():
    kp_good = Keypoint(x=100, y=200, confidence=0.8)
    kp_bad = Keypoint(x=0, y=0, confidence=0.1)
    assert kp_good.is_valid()
    assert not kp_bad.is_valid()
    assert not kp_bad.is_valid(min_confidence=0.3)


def test_keypoint_as_array():
    kp = Keypoint(x=50.0, y=75.0, confidence=0.9)
    arr = kp.as_array()
    assert arr.shape == (2,)
    assert arr[0] == 50.0
    assert arr[1] == 75.0


def test_pose_frame_get():
    pose = _make_pose()
    kp = pose.get(KP.LEFT_ANKLE)
    assert isinstance(kp, Keypoint)
    assert kp.confidence == 0.9


def test_pose_frame_get_xy():
    pose = _make_pose()
    xy = pose.get_xy(KP.NOSE)
    assert xy is not None
    assert xy.shape == (2,)


def test_pose_frame_get_xy_invalid():
    pose = _make_pose(conf=0.0)  # all keypoints invalid
    xy = pose.get_xy(KP.NOSE)
    assert xy is None


def test_pose_frame_midpoint():
    pose = _make_pose()
    mid = pose.midpoint(KP.LEFT_HIP, KP.RIGHT_HIP)
    assert mid is not None
    l = pose.get_xy(KP.LEFT_HIP)
    r = pose.get_xy(KP.RIGHT_HIP)
    expected = (l + r) / 2
    np.testing.assert_allclose(mid, expected)


def test_pose_frame_normalized():
    pose = _make_pose()
    norm = pose.normalized()
    for kp in norm.keypoints:
        if kp.is_valid():
            assert 0.0 <= kp.x <= 1.0
            assert 0.0 <= kp.y <= 1.0


def test_visible_keypoint_count():
    pose = _make_pose(conf=0.9)
    assert pose.visible_keypoint_count() == 17

    pose_bad = _make_pose(conf=0.1)
    assert pose_bad.visible_keypoint_count() == 0


def test_is_good_quality():
    pose = _make_pose()
    assert pose.is_good_quality()

    pose_low = _make_pose(conf=0.1)
    pose_low.detection_confidence = 0.1
    assert not pose_low.is_good_quality()


def test_kp_enum_coverage():
    assert len(KP) == 17
    assert KP.NOSE == 0
    assert KP.LEFT_ANKLE == 15
    assert KP.RIGHT_ANKLE == 16


def test_skeleton_edges_valid():
    for kp_a, kp_b in SKELETON_EDGES:
        assert isinstance(kp_a, KP)
        assert isinstance(kp_b, KP)


# ──────────────────────────────────────────
# Ingestion tests (test source, no hardware)
# ──────────────────────────────────────────

def _test_config():
    return {
        "camera": {
            "source": "test",
            "width": 640,
            "height": 480,
            "fps": 10,
            "use_gstreamer": False,
        },
        "output": {"show_debug_window": False},
        "logging": {"level": "WARNING", "log_to_file": False, "log_path": ""},
    }


def test_test_capture_opens():
    cap = _TestCapture(640, 480, 10)
    assert cap.isOpened()


def test_test_capture_reads_frame():
    cap = _TestCapture(640, 480, 30)  # 30fps for fast test
    ret, frame = cap.read()
    assert ret
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_ingestion_starts_and_stops():
    config = _test_config()
    ingestion = FrameIngestion(config)
    ingestion.start()
    assert ingestion.is_running

    # Collect a few frames
    frames = []
    for f in ingestion.frames(timeout=1.0):
        frames.append(f)
        if len(frames) >= 3:
            break

    ingestion.stop()
    assert not ingestion.is_running
    assert len(frames) == 3


def test_ingestion_frame_structure():
    config = _test_config()
    with FrameIngestion(config) as ingestion:
        for frame in ingestion.frames(timeout=1.0):
            assert isinstance(frame, CapturedFrame)
            assert frame.bgr.shape == (480, 640, 3)
            assert frame.width == 640
            assert frame.height == 480
            assert frame.frame_index >= 0
            assert frame.timestamp > 0
            break


def test_ingestion_frame_indices_increment():
    config = _test_config()
    indices = []
    with FrameIngestion(config) as ingestion:
        for frame in ingestion.frames(timeout=1.0):
            indices.append(frame.frame_index)
            if len(indices) >= 5:
                break

    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)  # all unique


def test_ingestion_context_manager():
    config = _test_config()
    with FrameIngestion(config) as ing:
        assert ing.is_running
    assert not ing.is_running


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
