import types
from pathlib import Path
import asr_bench
from tests.test_render import _whisper_result, _nim_result


def test_json_sanitize_nan_and_inf_to_none():
    out = asr_bench._json_sanitize(
        {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 1.5}
    )
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"] == 1.5


def test_json_sanitize_tuples_become_lists_recursively():
    out = asr_bench._json_sanitize({"segs": [(0.0, 1.0, "S0"), (1.0, 2.0, "S1")]})
    assert out["segs"] == [[0.0, 1.0, "S0"], [1.0, 2.0, "S1"]]
    assert isinstance(out["segs"][0], list)


def test_json_sanitize_passes_clean_values():
    val = {"x": 1, "y": "str", "z": [1, 2, {"w": None}], "b": True}
    assert asr_bench._json_sanitize(val) == val


def _cmd_args():
    return types.SimpleNamespace(
        models=["small", "large-v3-turbo"], device="cuda", compute_type="float16",
        batch_size=1, beam_size=5, vad_filter=True,
        nim_url="localhost:50051", nim_model="", nim_language="en-US",
    )


def test_reproducibility_command_basic():
    cmd = asr_bench._reproducibility_command(_cmd_args(), Path("/corpus"), [_whisper_result()])
    assert cmd.startswith("python asr_bench.py --corpus '/corpus' --models small,large-v3-turbo")
    assert "--device cuda" in cmd and "--compute-type float16" in cmd
    assert "--batch-size" not in cmd and "--beam-size" not in cmd and "--no-vad-filter" not in cmd


def test_reproducibility_command_nondefault_flags_and_nim():
    args = _cmd_args()
    args.batch_size = 8; args.beam_size = 3; args.vad_filter = False
    args.nim_model = "canary"
    cmd = asr_bench._reproducibility_command(args, Path("/c"), [_nim_result()])
    assert "--batch-size 8" in cmd and "--beam-size 3" in cmd and "--no-vad-filter" in cmd
    assert "--nim-url localhost:50051" in cmd and "--nim-model canary" in cmd
