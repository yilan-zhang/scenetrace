"""Dependency-free tracking and debounced spatial events.

These are observations from boxes, not semantic or physical ground truth.
"""
from dataclasses import dataclass
from math import hypot


@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass
class Track:
    id: int
    detection: Detection
    hits: int = 1
    missed: int = 0
    announced: bool = False

    @property
    def name(self):
        return f"{self.detection.label} #{self.id}"


def center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


class SceneAnalyzer:
    def __init__(self, confirm_frames=2, max_missing=3, dead_zone=0.06):
        if confirm_frames < 1 or max_missing < 1 or not 0 <= dead_zone < 1:
            raise ValueError("Invalid temporal settings")
        self.confirm_frames = confirm_frames
        self.max_missing = max_missing
        self.dead_zone = dead_zone
        self.tracks = {}
        self.next_id = 1
        self.relations = {}
        self.candidates = {}
        self.events = []

    def update(self, detections, frame, timestamp, width, height):
        start = len(self.events)
        previous = set(self.tracks)
        used_tracks, used_detections = set(), set()
        matches = []
        # Greedy one-to-one association: same label, sufficient overlap or nearby center.
        for tid, track in self.tracks.items():
            for index, detection in enumerate(detections):
                if track.detection.label != detection.label:
                    continue
                a, b = center(track.detection.box), center(detection.box)
                distance = hypot((a[0] - b[0]) / width, (a[1] - b[1]) / height)
                overlap = iou(track.detection.box, detection.box)
                if overlap >= 0.1 or distance <= 0.12:
                    matches.append((overlap - distance, tid, index))
        for _, tid, index in sorted(matches, reverse=True):
            if tid in used_tracks or index in used_detections:
                continue
            track = self.tracks[tid]
            track.hits = track.hits + 1 if track.missed == 0 else 1
            track.detection, track.missed = detections[index], 0
            used_tracks.add(tid)
            used_detections.add(index)
        for index, detection in enumerate(detections):
            if index not in used_detections:
                self.tracks[self.next_id] = Track(self.next_id, detection)
                used_tracks.add(self.next_id)
                self.next_id += 1
        for tid in previous - used_tracks:
            track = self.tracks[tid]
            track.missed += 1
            track.hits = 0
            if track.missed >= self.max_missing:
                if track.announced:
                    self._emit(frame, timestamp, "no_longer_detected", [tid], f"{track.name}: not detected for {self.max_missing} analyzed frames")
                del self.tracks[tid]
        visible = [t for t in self.tracks.values() if t.missed == 0]
        for track in visible:
            if not track.announced and track.hits >= self.confirm_frames:
                track.announced = True
                self._emit(frame, timestamp, "first_detected", [track.id], f"{track.name}: detection confirmed")
        confirmed = [t for t in visible if t.announced]
        active_pairs = set()
        for index, a in enumerate(confirmed):
            for b in confirmed[index + 1:]:
                if a.id > b.id:
                    a, b = b, a
                key = (a.id, b.id)
                active_pairs.add(key)
                dx = (center(a.detection.box)[0] - center(b.detection.box)[0]) / width
                relation = "left_of" if dx < -self.dead_zone else "right_of" if dx > self.dead_zone else "aligned"
                old_candidate, count = self.candidates.get(key, (None, 0))
                count = count + 1 if old_candidate == relation else 1
                self.candidates[key] = (relation, count)
                if count >= self.confirm_frames and self.relations.get(key) != relation:
                    old = self.relations.get(key)
                    self.relations[key] = relation
                    description = f"{a.name} {relation.replace('_', ' ')} {b.name}"
                    if old is not None:
                        description += f" (was {old.replace('_', ' ')})"
                    self._emit(frame, timestamp, "relation_observed" if old is None else "relation_changed", list(key), description)
        # Missing frames break the consecutive-observation debounce, but not remembered relations.
        self.candidates = {k: v for k, v in self.candidates.items() if k in active_pairs}
        self.relations = {k: v for k, v in self.relations.items() if all(t in self.tracks for t in k)}
        return visible, self.events[start:]

    def _emit(self, frame, timestamp, kind, ids, description):
        self.events.append({"frame": frame, "time_seconds": round(timestamp, 3), "kind": kind,
                            "track_ids": ids, "description": description})
