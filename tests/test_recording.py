"""Hardware-free synchronization, retention, and export acceptance tests."""
import csv
import json
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
from deployment.recording import ControlSample, EpisodeRecorder
from deployment.zed_capture import FramePacket, CaptureEvent, FramePairBroker
from deployment.export_dataset import action_chunk, export, segments
from deployment.yam_policy import YamInputs


def frame(side, sequence, timestamp):
    image = np.zeros((6, 9, 3), dtype=np.uint8)
    image[:, :, 0 if side == "left" else 2] = 240
    return FramePacket(side, 1 if side == "left" else 2, sequence, timestamp,
                       timestamp, timestamp, timestamp, image)


def control(tick, timestamp, valid=True):
    return ControlSample(tick, timestamp, tuple(range(14)), tuple(range(20, 34)),
                         "engaged", ((timestamp, timestamp+1),)*2,
                         ((timestamp+2, timestamp+3),)*2, (timestamp,)*2, valid)


class RecordingTests(unittest.TestCase):
    def test_broker_can_discard_warmup_then_start_at_zero(self):
        warmup, recorded = [], []
        broker = FramePairBroker(tolerance_ns=10, max_pending_per_side=4,
                                 callback=warmup.append)
        broker.set_callback(recorded.append, reset_counts=True)
        for side, seq, stamp in [("left", 0, 100), ("right", 0, 101),
                                 ("left", 1, 133), ("right", 1, 134)]:
            broker.add(frame(side, seq, stamp))
        broker.close()
        self.assertEqual(warmup, [])
        self.assertEqual([event.index for event in recorded], [0, 1])

    def test_causal_match_skips_invalid_and_never_uses_earlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, task="place vial", min_free_gb=0)
            recorder.add_control(control(0, 90))
            recorder.add_control(control(1, 101, False))
            recorder.add_control(control(2, 105))
            event = CaptureEvent(0, "pair", frame("left", 0, 100), frame("right", 0, 100))
            self.assertEqual(recorder._match(event).tick, 2)
            recorder.close()

    def test_broker_orphan_does_not_shift_next_frame(self):
        events = []
        broker = FramePairBroker(tolerance_ns=10, max_pending_per_side=4, callback=events.append)
        for side, seq, stamp in [("left", 0, 100), ("left", 1, 133),
                                 ("right", 1, 134), ("left", 2, 166), ("right", 2, 167)]:
            broker.add(frame(side, seq, stamp))
        broker.close()
        self.assertEqual([e.kind for e in events], ["left_orphan", "pair", "pair"])
        self.assertEqual(events[1].left.sequence, events[1].right.sequence)
        broker.add(frame("left", 3, 100))
        self.assertIn("not_monotonic", events[-1].reason)

    def test_three_camera_broker_requires_one_frame_from_every_view(self):
        events = []
        broker = FramePairBroker(tolerance_ns=10, max_pending_per_side=4,
                                 callback=events.append,
                                 sides=("left", "right", "overhead"))
        for side, seq, stamp in [("left", 0, 100), ("right", 0, 101), ("overhead", 0, 102),
                                 ("left", 1, 133), ("right", 1, 134), ("overhead", 1, 135)]:
            broker.add(frame(side, seq, stamp))
        broker.close()
        self.assertEqual([event.kind for event in events], ["pair", "pair"])
        self.assertEqual(set(events[0].frames), {"left", "right", "overhead"})
        self.assertEqual(events[0].pair_skew_ns, 2)

    def test_raw_orphan_retention_rgb_and_gap_export(self):
        import h5py
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, task="place vial", min_free_gb=0)
            base = time.monotonic_ns()
            for i in range(5):
                timestamp = base + i * 33_333_333
                recorder.add_control(control(i, timestamp + 1_000_000))
                recorder.add_capture(CaptureEvent(i, "left_orphan" if i == 2 else "pair",
                  frame("left", i, timestamp), None if i == 2 else frame("right", i, timestamp),
                  "right_missing" if i == 2 else ""))
            path = recorder.close()
            self.assertNotEqual(path.suffix, ".partial")
            with (path / "data.csv").open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5)
            self.assertTrue((path / rows[2]["left_image"]).exists())
            self.assertFalse(rows[2]["right_image"])
            self.assertEqual(rows[2]["valid"], "0")
            self.assertEqual([len(s) for s in segments(path)], [2, 2])
            paths = export(path, Path(tmp) / "hdf5")
            self.assertEqual(len(paths), 2)
            with paths[0].with_suffix(".csv").open() as f:
                header = next(csv.reader(f))
            self.assertEqual(len(header), 16)
            self.assertEqual(header[2], "left_q1")
            with h5py.File(paths[0]) as f:
                self.assertEqual(f["additional_info/frequency"][()], 30)
                self.assertEqual(f["action/left_arm_joint_states"][0, 0], 20)
                self.assertEqual(f["state/left_arm_joint_states"][0, 0], 0)
                self.assertGreater(f["vision/cam_left_wrist/colors"][0, 0, 0, 0], 220)
            self.assertEqual(len((path / "controls.jsonl").read_text().splitlines()), 5)

    def test_missing_pair_time_splits_even_without_orphan(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, task="test", min_free_gb=0)
            base = time.monotonic_ns()
            for i in (0, 1, 3, 4):
                stamp = base + i*33_333_333
                recorder.add_control(control(i, stamp+1))
                recorder.add_capture(CaptureEvent(i, "pair", frame("left", i, stamp), frame("right", i, stamp)))
            path = recorder.close()
            self.assertEqual([len(s) for s in segments(path)], [2, 2])

    def test_three_camera_record_export_and_policy_base_view(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            metadata = {"capture": {"camera_names":
                        ["cam_left_wrist", "cam_right_wrist", "cam_overhead"]}}
            recorder = EpisodeRecorder(tmp, task="test", metadata=metadata, min_free_gb=0,
                                       camera_sides=("left", "right", "overhead"))
            stamp = time.monotonic_ns()
            recorder.add_control(control(0, stamp + 1))
            recorder.add_capture(CaptureEvent(0, "pair", frame("left", 0, stamp),
                                 frame("right", 0, stamp), overhead=frame("overhead", 0, stamp)))
            path = recorder.close()
            outputs = export(path, Path(tmp) / "hdf5")
            with outputs[0].with_suffix(".csv").open() as f:
                self.assertEqual(len(next(csv.reader(f))), 17)
            with h5py.File(outputs[0]) as f:
                self.assertIn("cam_overhead", f["vision"])
            result = YamInputs()({"state": np.zeros(14),
                                  "images/left": np.zeros((6, 9, 3), dtype="u1"),
                                  "images/right": np.zeros((6, 9, 3), dtype="u1"),
                                  "images/base": np.ones((6, 9, 3), dtype="u1")})
            self.assertTrue(result["image_mask"]["base_0_rgb"])
            self.assertEqual(result["image"]["base_0_rgb"][0, 0, 0], 1)

    def test_failure_keeps_partial_and_refuses_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, task="test", min_free_gb=0)
            recorder.fail("injected disk failure")
            path = recorder.close()
            self.assertEqual(path.suffix, ".partial")
            with self.assertRaises(ValueError):
                list(segments(path))

    def test_image_write_failure_marks_partial(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch("cv2.imwrite", return_value=False):
                recorder = EpisodeRecorder(tmp, task="test", min_free_gb=0)
                stamp = time.monotonic_ns()
                recorder.add_control(control(0, stamp+1))
                recorder.add_capture(CaptureEvent(0, "pair", frame("left", 0, stamp), frame("right", 0, stamp)))
                path = recorder.close()
            self.assertEqual(path.suffix, ".partial")
            self.assertIn("image write failed", json.loads((path / "metadata.json").read_text())["errors"][0])

    def test_repeated_active_orphans_invalidate_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, task="test", min_free_gb=0)
            stamp = time.monotonic_ns()
            recorder.add_control(control(0, stamp+1))
            recorder.add_capture(CaptureEvent(0, "pair", frame("left", 0, stamp), frame("right", 0, stamp)))
            for i in range(1, 6):
                recorder.add_capture(CaptureEvent(i, "left_orphan", frame("left", i, stamp+i*33_333_333), None, "missing_right"))
            path = recorder.close()
            self.assertEqual(path.suffix, ".partial")
            self.assertIn("repeated invalid", json.loads((path / "metadata.json").read_text())["errors"][0])

    def test_outside_skew_is_not_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(tmp, task="test", min_free_gb=0, max_skew_ms=1)
            recorder.add_control(control(0, 2_000_001))
            event = CaptureEvent(0, "pair", frame("left", 0, 1), frame("right", 0, 1))
            self.assertIsNone(recorder._match(event))
            recorder.close()

    def test_chunks_padding_and_mask(self):
        actions = np.arange(42).reshape(3, 14)
        chunk, mask = action_chunk(actions, 1, 4)
        np.testing.assert_equal(mask, [True, True, False, False])
        np.testing.assert_equal(chunk[-1], actions[-1])

    def test_policy_missing_camera_mask(self):
        result = YamInputs()({"state": np.zeros(14), "images/left": np.zeros((6,9,3), dtype="u1"),
                              "images/right": np.zeros((6,9,3), dtype="u1")})
        self.assertFalse(result["image_mask"]["base_0_rgb"])
        self.assertTrue(result["image_mask"]["left_wrist_0_rgb"])

if __name__ == "__main__":
    unittest.main()
