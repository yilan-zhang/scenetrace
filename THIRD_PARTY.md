# Third-party materials

- **Ultralytics YOLO**: version 8.3.221. Source: https://github.com/ultralytics/ultralytics/tree/v8.3.221 . Upstream license: AGPL-3.0. Used for pretrained inference; no training or fine-tuning is performed here.
- **YOLOv8n weights**: downloaded by Ultralytics from https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt . Weights are not included in the submission archive.
- **Example photo**: `ultralytics/assets/bus.jpg`, distributed in Ultralytics 8.3.221. Source: https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/assets/bus.jpg . Copied unchanged as `examples/street.jpg`; not photographed or collected by this project's author. The upstream distribution's license and notices apply to this asset.
- **Derived demo**: `examples/controlled_clip.mp4` resizes/translates the example photo and inserts a blank interval. It is an explicitly synthetic temporal test, not independently collected video. The script in `scripts/make_demo.py` documents all transformations.
- **Other dependencies**: OpenCV, NumPy, PyTorch, and the remaining packages retain their upstream licenses; their code is installed separately and not bundled in the submission archive.

No private research dataset, company repository, or internal system is used.
