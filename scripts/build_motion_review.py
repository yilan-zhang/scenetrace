"""Rebuild selected evidence panels from saved detections, without rerunning YOLO."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'examples/natural_motion'
report = json.loads((BASE / 'result/report.json').read_text())
frames = {f['frame']: f for f in report['frames']}
cap = cv2.VideoCapture(str(BASE / 'input.mp4'))


def panel(index, ids, title):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'Cannot decode frame {index}')
    for detection in frames[index]['detections']:
        if detection['track_id'] not in ids:
            continue
        x1, y1, x2, y2 = map(int, detection['box_xyxy'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 245, 185), 3)
        label = f"person #{detection['track_id']}"
        cv2.putText(frame, label, (x1 + 4, max(26, y1 + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 5, cv2.LINE_AA)
        cv2.putText(frame, label, (x1 + 4, max(26, y1 + 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 245, 185), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (640, 360))
    result = np.full((420, 640, 3), (23, 20, 16), dtype=np.uint8)
    result[60:] = frame
    cv2.putText(result, title, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(result, f"{index / report['source_fps']:.2f}s / original frame {index}", (16, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (160, 175, 170), 1, cv2.LINE_AA)
    return result


cases = [
 ('exit_check.jpg', (50, [1], 'Before: foreground person #1'), (70, [1], 'After: #1 loss confirmed')),
 ('relation_check.jpg', (100, [6, 9], 'Before: #6 right of #9'), (125, [6, 9], 'After: #6 left of #9')),
 ('identity_switch.jpg', (75, [5], 'Track #5: beige top, approaching camera'), (225, [5], 'Track #5: blue shirt, walking away')),
]
for filename, left, right in cases:
    cv2.imwrite(str(BASE / filename), np.hstack([panel(*left), panel(*right)]))
cap.release()
print('Built 3 paired evidence panels from saved detections.')
