import asr_bench


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
