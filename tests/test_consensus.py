from consensus import vote_window_texts, build_consensus_clip
from engines.base import Cue


def test_vote_window_majority():
    texts = [
        "the quick brown fox",
        "the quick brown dog",
        "the quick brown fox",
    ]
    out, agree, flags = vote_window_texts(texts, min_ratio=0.5)
    assert "quick" in out
    assert "brown" in out
    assert agree > 0.5
    assert any("dog" in f or "fox" in f for f in flags) or "fox" in out


def test_vote_window_empty():
    out, agree, flags = vote_window_texts([])
    assert out == ""
    assert agree == 0.0


def test_build_consensus_clip_windows():
    sources = {
        "a": [Cue(0, 10, "hello world today")],
        "b": [Cue(0, 10, "hello world today")],
    }
    res = build_consensus_clip(30.0, sources, window=15, overlap=0, min_agreement=0.5)
    assert res.cues
    assert res.mean_agreement >= 0.5
