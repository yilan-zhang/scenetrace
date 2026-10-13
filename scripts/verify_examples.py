"""Verify saved real-inference outputs without downloading or rerunning a model."""
import csv
import json
from pathlib import Path
import cv2

root = Path(__file__).resolve().parents[1] / "examples"
for name in ("image_result", "video_result", "empty_result", "natural_motion/result"):
    folder = root / name
    report = json.loads((folder / "report.json").read_text())
    assert len(report["frames"]) == report["analyzed_frames"]
    assert sum(not f["detections"] for f in report["frames"]) == report["empty_frames"]
    with (folder / "events.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(report["events"])
    assert cv2.imread(str(folder / "preview.jpg")) is not None
    frame_ids = {f["frame"] for f in report["frames"]}
    for event in report["events"]:
        assert event["frame"] in frame_ids
        assert cv2.imread(str(folder / event["evidence"])) is not None
    print(f"PASS {name}: {report['analyzed_frames']} frames, {len(report['events'])} events")
image = json.loads((root / "image_result/report.json").read_text())
assert image["peak_visible_counts"].get("bus", 0) >= 1
assert image["peak_visible_counts"].get("person", 0) >= 1
video = json.loads((root / "video_result/report.json").read_text())
assert 0 < video["empty_frames"] < video["analyzed_frames"]
assert any(e["kind"] == "no_longer_detected" for e in video["events"])
assert any(e["kind"] == "first_detected" and e["time_seconds"] >= 6 for e in video["events"])
cap = cv2.VideoCapture(str(root / "video_result/annotated.mp4"))
count = 0
while True:
    ok, _ = cap.read()
    if not ok:
        break
    count += 1
cap.release()
assert count == video["analyzed_frames"]
empty = json.loads((root / "empty_result/report.json").read_text())
assert empty["empty_frames"] == 1 and empty["events"] == []
print("PASS decoded video length, disappearance/reappearance, and empty-image behavior")

# Check natural-motion media, the complete event distribution, and evidence used in the review.
from collections import Counter
import hashlib
natural = root / "natural_motion"
assert hashlib.sha256((natural / "source.webm").read_bytes()).hexdigest() == "2834fbc411cc59f06a63f9ac7ce9ba5b748b67242d1b36a4b18043a070a7966a"
r = json.loads((natural / "result/report.json").read_text())
assert r["analyzed_frames"] == 72
assert Counter(e["kind"] for e in r["events"]) == {"first_detected": 26, "no_longer_detected": 19, "relation_observed": 92, "relation_changed": 3}
assert any(e["kind"] == "relation_changed" and e["track_ids"] == [6, 9] and e["frame"] == 125 for e in r["events"])
for name, expected in [("input.mp4", 356), ("result/annotated.mp4", 72)]:
    cap = cv2.VideoCapture(str(natural / name))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    assert count == expected, (name, count)
for name in ("exit_check.jpg", "relation_check.jpg", "identity_switch.jpg"):
    assert cv2.imread(str(natural / name)).shape[:2] == (420, 1280)
print("PASS natural-motion provenance, decoded frames, event counts, and review panels")
