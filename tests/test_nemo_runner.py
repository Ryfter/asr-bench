# tests/test_nemo_runner.py
"""Torch-free helpers of nemo_runner. The heavy run_nemo path is live-validated
on the RTX 5090, not unit-tested (no torch/nemo in the core venv)."""
import nemo_runner


def test_arg_parser_required_and_defaults():
    ns = nemo_runner.build_arg_parser().parse_args(
        ["--audio", "a.wav", "--model", "nvidia/parakeet-tdt-0.6b-v2"])
    assert ns.audio == "a.wav"
    assert ns.model == "nvidia/parakeet-tdt-0.6b-v2"
    assert ns.device == "cuda"
    assert ns.language == "en"
    assert ns.chunk_len_secs == 40.0
    assert ns.max_new_tokens >= 256        # generous default so chunks don't truncate


def test_looks_like_salm():
    assert nemo_runner._looks_like_salm("nvidia/canary-qwen-2.5b") is True
    assert nemo_runner._looks_like_salm("nvidia/parakeet-tdt-0.6b-v2") is False


def test_segments_to_json_from_dicts():
    segs = [{"start": 0.0, "end": 2.0, "segment": "hello there"},
            {"start": 2.0, "end": 4.0, "text": "hi back"}]
    out = nemo_runner._segments_to_json(segs)
    assert out == [{"start": 0.0, "end": 2.0, "text": "hello there"},
                   {"start": 2.0, "end": 4.0, "text": "hi back"}]


def test_words_to_json_from_dicts():
    words = [{"word": "hello", "start": 0.0, "end": 0.5}]
    out = nemo_runner._words_to_json(words)
    assert out == [{"word": "hello", "start": 0.0, "end": 0.5}]


def test_segments_to_json_empty():
    assert nemo_runner._segments_to_json([]) == []
    assert nemo_runner._words_to_json(None) == []
