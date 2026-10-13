"""Create a reproducible controlled clip from Ultralytics' bundled example photo.

The clip is an edited still-image test, not natural motion or a tracking benchmark.
"""
from pathlib import Path
import shutil
import cv2
import numpy as np
import ultralytics

root = Path(__file__).resolve().parents[1]
out = root / "examples"
out.mkdir(exist_ok=True)
source = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
if not source.is_file():
    raise SystemExit("Bundled bus.jpg missing; reinstall the pinned ultralytics package.")
shutil.copy2(source, out / "street.jpg")
photo = cv2.imread(str(source))
photo = cv2.resize(photo, (480, 640))
writer = cv2.VideoWriter(str(out / "controlled_clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (480, 640))
if not writer.isOpened():
    raise SystemExit("Video encoder unavailable")
# Original photo -> translated photo -> blank -> restored photo.
# The blank interval checks missing-detection handling; transitions are intentionally synthetic.
for index in range(80):
    if index < 20:
        frame = photo.copy()
    elif index < 40:
        offset = int((index - 20) * 2)
        frame = cv2.warpAffine(photo, np.float32([[1, 0, offset], [0, 1, 0]]), (480, 640))
    elif index < 60:
        frame = np.full_like(photo, 32)
    else:
        frame = photo.copy()
    writer.write(frame)
writer.release()
cv2.imwrite(str(out / "empty.png"), np.full((640, 480, 3), 32, dtype=np.uint8))
print("Created street.jpg, controlled_clip.mp4, empty.png")
