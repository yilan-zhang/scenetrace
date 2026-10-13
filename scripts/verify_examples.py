"""Verify saved real-inference outputs without downloading or rerunning a model."""
import csv
import json
from pathlib import Path
import cv2

root = Path(__file__).resolve().parents[1] / "examples"
for name in ("image_result", "video_result", "empty_result"):
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
