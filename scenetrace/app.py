"""Local image/video inference, annotated evidence, and machine-readable events."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import textwrap

from .logic import Detection, SceneAnalyzer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def annotate(cv2, np, frame, tracks, analyzer, timestamp, frame_index, counts):
    height, width = frame.shape[:2]
    panel_width = 390
    canvas_height = max(height + 84, 720)
    canvas = np.full((canvas_height, width + panel_width, 3), (23, 20, 16), dtype=np.uint8)
    canvas[84:84 + height, :width] = frame
    white, muted, accent = (241, 241, 235), (168, 156, 141), (162, 222, 111)

    def write(text, x, y, size=0.5, color=white, thickness=1):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

    write("SCENETRACE", 22, 34, 0.9, white, 2)
    write("Detection-backed video observations", 23, 61, 0.45, muted)
    write(f"{timestamp:06.2f}s   /   FRAME {frame_index:04d}", width + 22, 34, 0.6, accent)
    write("OBSERVATIONS, NOT GROUND TRUTH", width + 22, 59, 0.4, muted)
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.detection.box)
        color = [(162, 222, 111), (239, 192, 111), (166, 157, 251), (116, 212, 249)][(track.id - 1) % 4]
        cv2.rectangle(canvas, (x1, y1 + 84), (x2, y2 + 84), color, 2)
        label = f"{track.name}  {track.detection.confidence:.2f}"
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
        top = max(84, y1 + 60)
        left = max(0, min(x1, width - label_width - 12))
        cv2.rectangle(canvas, (left, top), (min(width, left + label_width + 12), top + 24), color, -1)
        write(label, left + 5, top + 17, 0.48, (25, 25, 25))
    x, y = width + 22, 107
    write("IN THIS FRAME", x, y, 0.48, muted)
    y += 34
    write(f"{len(tracks)} objects", x, y, 0.95, white, 2)
    y += 28
    for line in textwrap.wrap(", ".join(f"{n} {label}" for label, n in sorted(counts.items())) or "No objects detected", 43):
        write(line, x, y, 0.43, accent)
        y += 21
    y += 30
    write("SPATIAL RELATIONS", x, y, 0.48, muted)
    y += 26
    visible_ids = {t.id for t in tracks}
    shown = 0
    for (a, b), relation in analyzer.relations.items():
        if a not in visible_ids or b not in visible_ids:
            continue
        text = f"{analyzer.tracks[a].name} {relation.replace('_', ' ')} {analyzer.tracks[b].name}"
        for line in textwrap.wrap(text, 40):
            write(line, x, y, 0.43)
            y += 20
        shown += 1
        if shown == 4:
            break
    if not shown:
        write("No confirmed relation", x, y, 0.43, muted)
        y += 20
    y += 30
    write("RECENT EVENTS", x, y, 0.48, muted)
    y += 26
    for event in analyzer.events[-3:]:
        write(f"{event['time_seconds']:.2f}s  {event['kind'].replace('_', ' ')}", x, y, 0.43, accent)
        y += 20
        for line in textwrap.wrap(event["description"], 43):
            if y > canvas_height - 42:
                break
            write(line, x, y, 0.4)
            y += 19
        y += 13
    write("Full timeline: events.csv + report.json", x, canvas_height - 19, 0.4, muted)
    # Common encoders require even dimensions.
    return canvas[:canvas.shape[0] // 2 * 2, :canvas.shape[1] // 2 * 2]


def parser():
    result = argparse.ArgumentParser(description="Detect objects and export evidence-backed spatial event logs.")
    result.add_argument("source", type=Path, help="Local image or video")
    result.add_argument("--output", type=Path, default=Path("outputs/run"), help="New or empty output directory")
    result.add_argument("--model", default="yolov8n.pt", help="Pretrained Ultralytics detection weights")
    result.add_argument("--classes", default="", help="Comma-separated labels, e.g. person,bus; default: all")
    result.add_argument("--conf", type=float, default=0.35)
    result.add_argument("--stride", type=int, default=3, help="Analyze every Nth video frame")
    result.add_argument("--confirm", type=int, default=2, help="Consecutive analyzed frames to confirm an observation")
    result.add_argument("--max-missing", type=int, default=3, help="Missed analyzed frames before a loss event")
    result.add_argument("--device", default="cpu", help="cpu, mps, or a CUDA device index")
    result.add_argument("--max-frames", type=int, default=0, help="Stop after N analyzed frames; 0: entire video")
    return result


def run(args):
    if not args.source.is_file():
        raise ValueError(f"Input file does not exist: {args.source}")
    if not 0 < args.conf <= 1 or min(args.stride, args.confirm, args.max_missing) < 1 or args.max_frames < 0:
        raise ValueError("Confidence must be in (0, 1]; temporal settings must be positive; max-frames cannot be negative.")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output}. Choose a new directory.")
    import cv2
    import numpy as np
    from ultralytics import YOLO

    is_image = args.source.suffix.lower() in IMAGE_EXTENSIONS
    capture = None
    if is_image:
        initial = cv2.imread(str(args.source))
        if initial is None:
            raise ValueError("Unable to decode input image")
        fps, stride = 1.0, 1
    else:
        capture = cv2.VideoCapture(str(args.source))
        if not capture.isOpened():
            raise ValueError("Unable to decode input video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError("Video has no valid frame rate; re-encode it before analysis.")
        stride = args.stride
    model = YOLO(args.model)
    wanted = {label.strip() for label in args.classes.split(",") if label.strip()}
    unknown = wanted - set(model.names.values())
    if unknown:
        if capture is not None:
            capture.release()
        raise ValueError(f"Unknown model classes: {', '.join(sorted(unknown))}")
    classes = [index for index, label in model.names.items() if label in wanted] if wanted else None
    args.output.mkdir(parents=True, exist_ok=True)
    evidence_dir = args.output / "evidence"
    evidence_dir.mkdir()
    analyzer = SceneAnalyzer(1 if is_image else args.confirm, args.max_missing)
    frame_index, analyzed, empty_frames = 0, 0, 0
    writer = None
    peaks = Counter()
    frame_records = []
    preview_saved = False
    try:
        while True:
            if is_image:
                if frame_index:
                    break
                frame = initial
            else:
                success, frame = capture.read()
                if not success:
                    break
            current = frame_index
            frame_index += 1
            if current % stride:
                continue
            timestamp = current / fps
            prediction = model.predict(frame, conf=args.conf, classes=classes, max_det=12,
                                       device=args.device, verbose=False)[0]
            detections = [Detection(model.names[int(label)], float(conf), tuple(map(float, box)))
                          for box, conf, label in zip(prediction.boxes.xyxy.cpu().tolist(),
                                                     prediction.boxes.conf.cpu().tolist(),
                                                     prediction.boxes.cls.cpu().tolist())]
            height, width = frame.shape[:2]
            tracks, new_events = analyzer.update(detections, current, timestamp, width, height)
            counts = Counter(d.label for d in detections)
            peaks |= counts
            empty_frames += int(not detections)
            annotated = annotate(cv2, np, frame, tracks, analyzer, timestamp, current, counts)
            if new_events:
                evidence_name = f"evidence/frame_{current:06d}.jpg"
                if not cv2.imwrite(str(args.output / evidence_name), annotated):
                    raise OSError("Unable to save evidence image")
                for event in new_events:
                    event["evidence"] = evidence_name
            if not preview_saved or (analyzer.relations and not (args.output / "relations.jpg").exists()):
                cv2.imwrite(str(args.output / "preview.jpg"), annotated)
                if analyzer.relations:
                    cv2.imwrite(str(args.output / "relations.jpg"), annotated)
                preview_saved = True
            if is_image:
                cv2.imwrite(str(args.output / "annotated.jpg"), annotated)
            else:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps / stride, (annotated.shape[1], annotated.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("MP4 encoder unavailable in this OpenCV installation")
                writer.write(annotated)
            frame_records.append({"frame": current, "time_seconds": round(timestamp, 3), "counts": dict(counts),
                                  "detections": [{"track_id": t.id, "label": t.detection.label,
                                                  "confidence": round(t.detection.confidence, 4),
                                                  "box_xyxy": list(t.detection.box)} for t in tracks]})
            analyzed += 1
            if analyzed % 20 == 0:
                print(f"Analyzed {analyzed} frames | {timestamp:.2f}s | {len(analyzer.events)} events", flush=True)
            if args.max_frames and analyzed >= args.max_frames:
                break
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
    if analyzed == 0:
        raise ValueError("Input contains no decodable frames")
    report = {"source": args.source.name, "model": Path(args.model).name,
              "settings": {"confidence": args.conf, "classes": sorted(wanted) or "all", "stride": stride,
                           "confirm_frames": analyzer.confirm_frames, "max_missing": args.max_missing,
                           "dead_zone_fraction": analyzer.dead_zone, "max_detections": 12, "device": args.device},
              "source_fps": fps, "analyzed_frames": analyzed, "empty_frames": empty_frames,
              "peak_visible_counts": dict(peaks), "events": analyzer.events, "frames": frame_records,
              "interpretation": "Detection observations only. A missing detection does not prove a physical disappearance. Left/right uses 2D box centers."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "events.csv").open("w", newline="") as stream:
        fields = ["frame", "time_seconds", "kind", "track_ids", "description", "evidence"]
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(analyzer.events)
    lines = ["# SceneTrace run", "", f"Input: `{args.source.name}`", "",
             f"Analyzed **{analyzed} frames**, including **{empty_frames} without detections**; logged **{len(analyzer.events)} events**.",
             "", "![Annotated preview](preview.jpg)", "", "## Event timeline", ""]
    if not analyzer.events:
        lines.append("No confirmed events. See report.json for per-frame detections.")
    for event in analyzer.events:
        lines.append(f"- **{event['time_seconds']:.2f}s** — {event['description']} ([evidence]({event['evidence']}))")
    lines += ["", "Events reflect model observations, not verified scene changes. End of file does not imply that visible objects disappeared.", ""]
    (args.output / "SUMMARY.md").write_text("\n".join(lines))
    print(f"Done: {analyzed} frames, {len(analyzer.events)} events, {empty_frames} empty frames\nResults: {args.output.resolve()}")
    return report


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"SceneTrace: {exc}", file=sys.stderr)
        raise SystemExit(1)
