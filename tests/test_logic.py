from scenetrace.logic import Detection, SceneAnalyzer, iou


def detection(x, label="cup", confidence=0.9):
    return Detection(label, confidence, (x, 20, x + 20, 60))


def update(analyzer, ds, frame):
    return analyzer.update(ds, frame, frame / 10, 200, 100)


def test_empty_input():
    analyzer = SceneAnalyzer()
    assert update(analyzer, [], 0) == ([], [])
    assert analyzer.events == []


def test_short_false_positive_never_announced():
    analyzer = SceneAnalyzer()
    update(analyzer, [detection(20)], 0)
    for frame in range(1, 5):
        update(analyzer, [], frame)
    assert analyzer.events == []


def test_dropout_tolerance_and_confirmed_loss():
    analyzer = SceneAnalyzer()
    update(analyzer, [detection(20)], 0)
    tracks, events = update(analyzer, [detection(22)], 1)
    assert events[0]["kind"] == "first_detected"
    tid = tracks[0].id
    update(analyzer, [], 2)
    tracks, _ = update(analyzer, [detection(24)], 3)
    assert tracks[0].id == tid
    for frame in range(4, 7):
        update(analyzer, [], frame)
    losses = [e for e in analyzer.events if e["kind"] == "no_longer_detected"]
    assert len(losses) == 1
    assert losses[0]["frame"] == 6


def test_relation_change_requires_persistence():
    analyzer = SceneAnalyzer(confirm_frames=2)
    for frame in range(3):
        update(analyzer, [detection(60), detection(100, "bottle")], frame)
    assert analyzer.events[-1]["description"] == "cup #1 left of bottle #2"
    # Move gradually so association stays unambiguous.
    update(analyzer, [detection(80), detection(100, "bottle")], 3)
    update(analyzer, [detection(100), detection(100, "bottle")], 4)
    update(analyzer, [detection(120), detection(100, "bottle")], 5)
    assert not any(e["kind"] == "relation_changed" for e in analyzer.events)
    update(analyzer, [detection(120), detection(100, "bottle")], 6)
    assert analyzer.events[-1]["kind"] == "relation_changed"
    assert "right of" in analyzer.events[-1]["description"]


def test_consecutive_confirmation_resets_after_gap():
    analyzer = SceneAnalyzer(confirm_frames=2)
    update(analyzer, [detection(20)], 0)
    update(analyzer, [], 1)
    assert update(analyzer, [detection(20)], 2)[1] == []
    assert update(analyzer, [detection(20)], 3)[1][0]["kind"] == "first_detected"


def test_one_to_one_and_class_consistency():
    analyzer = SceneAnalyzer(confirm_frames=1)
    tracks, _ = update(analyzer, [detection(20), detection(40)], 0)
    assert len({t.id for t in tracks}) == 2
    tracks, _ = update(analyzer, [detection(21, "bottle")], 1)
    assert tracks[0].id == 3


def test_box_overlap_and_degenerate_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0
