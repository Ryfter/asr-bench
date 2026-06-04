import asr_bench
from asr_bench import Cue


def test_build_windows_tiles_with_overlap():
    # duration 60s, window 25s, overlap 5s -> stride 20s
    wins = asr_bench.build_windows(60.0, window=25.0, overlap=5.0)
    assert wins[0] == (0.0, 25.0)
    assert wins[1] == (20.0, 45.0)
    assert wins[2] == (40.0, 60.0)          # last clamped to duration
    assert wins[-1][1] == 60.0


def test_build_windows_short_clip_single_window():
    wins = asr_bench.build_windows(10.0, window=25.0, overlap=5.0)
    assert wins == [(0.0, 10.0)]


def test_collect_window_text_includes_overlapping_cues():
    cues = [Cue(0.0, 3.0, "alpha"), Cue(3.0, 22.0, "beta"), Cue(22.0, 40.0, "gamma")]
    # window [20,45] overlaps beta (ends 22>20) and gamma
    text = asr_bench.collect_window_text(cues, 20.0, 45.0)
    assert "beta" in text and "gamma" in text
    assert "alpha" not in text
