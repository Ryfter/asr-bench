import json
from pathlib import Path

import pytest
import asr_compare


def _doc(stem="run", *, models=None, corpus="test-corpus", config=None,
         schema_version=1):
    """Build a minimal schema_version-1 sidecar dict (already labeled)."""
    doc = {
        "schema_version": schema_version,
        "run": {
            "corpus": corpus,
            "config": config or {"device": "cuda", "compute_type": "float16",
                                 "beam_size": 5, "vad_filter": True, "batch_size": 1},
        },
        "models": models if models is not None else [],
        "_source_label": stem,
    }
    return doc


def _model(model_id="m", display=None, *, wer=0.10, mer=0.09, wil=0.12,
           rtfx=60.0, clips=None):
    return {
        "model_id": model_id,
        "display": display or model_id,
        "aggregates": {"avg_wer": wer, "avg_mer": mer, "avg_wil": wil,
                       "aggregate_rtfx": rtfx, "peak_vram_bytes": None},
        "clips": clips if clips is not None else [],
    }


def test_load_valid_v1(tmp_path):
    p = tmp_path / "20260606-120000.json"
    p.write_text(json.dumps(_doc()), encoding="utf-8")
    doc = asr_compare.load_results_json(p)
    assert doc is not None
    assert doc["schema_version"] == 1
    assert doc["_source_label"] == "20260606-120000"


def test_load_wrong_schema_version_returns_none(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(_doc(schema_version=2)), encoding="utf-8")
    assert asr_compare.load_results_json(p) is None
    assert "schema_version" in capsys.readouterr().err


def test_load_missing_file_returns_none(tmp_path, capsys):
    assert asr_compare.load_results_json(tmp_path / "nope.json") is None
    assert "skipping" in capsys.readouterr().err


def test_join_shared_model_collects_both_values():
    a = _doc("a", models=[_model("big", "Big", wer=0.10, rtfx=60.0)])
    b = _doc("b", models=[_model("big", "Big", wer=0.08, rtfx=70.0)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    row = [m for m in rep["models"] if m["model_id"] == "big"][0]
    assert row["values"]["wer"] == [0.10, 0.08]
    assert row["values"]["rtfx"] == [60.0, 70.0]
    assert row["present_in"] == [0, 1]
    assert row["status"] == "both"


def test_model_only_in_baseline_is_removed():
    a = _doc("a", models=[_model("old", "Old"), _model("keep", "Keep")])
    b = _doc("b", models=[_model("keep", "Keep")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    old = [m for m in rep["models"] if m["model_id"] == "old"][0]
    assert old["status"] == "removed"
    assert old["values"]["wer"] == [0.10, None]


def test_model_only_in_candidate_is_added():
    a = _doc("a", models=[_model("keep", "Keep")])
    b = _doc("b", models=[_model("keep", "Keep"), _model("new", "New")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    new = [m for m in rep["models"] if m["model_id"] == "new"][0]
    assert new["status"] == "added"
    assert new["values"]["wer"] == [None, 0.10]


def test_der_metric_absent_when_no_clip_der():
    a = _doc("a", models=[_model("m")])
    b = _doc("b", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert "der" not in rep["metrics"]


def test_der_metric_present_and_averaged_from_clips():
    clips = [{"audio": "x.mp4", "der": 0.10, "num_speakers": 2},
             {"audio": "y.mp4", "der": 0.20, "num_speakers": 2}]
    a = _doc("a", models=[_model("m", clips=clips)])
    b = _doc("b", models=[_model("m", clips=[{"audio": "x.mp4", "der": None,
                                              "num_speakers": None}])])
    rep = asr_compare.compare_runs([a, b], mode="matrix")
    assert "der" in rep["metrics"]
    row = rep["models"][0]
    assert row["values"]["der"][0] == 0.15      # (0.10 + 0.20) / 2
    assert row["values"]["der"][1] is None      # all-null clips -> None


def test_model_union_preserves_first_seen_order():
    a = _doc("a", models=[_model("z"), _model("a")])
    b = _doc("b", models=[_model("a"), _model("q")])
    rep = asr_compare.compare_runs([a, b], mode="matrix")
    assert [m["model_id"] for m in rep["models"]] == ["z", "a", "q"]


def test_deltas_candidate_minus_baseline():
    a = _doc("a", models=[_model("m", wer=0.10, rtfx=60.0)])
    b = _doc("b", models=[_model("m", wer=0.08, rtfx=70.0)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    d = rep["models"][0]["deltas"]
    assert round(d["wer"], 4) == -0.02
    assert round(d["rtfx"], 4) == 10.0


def test_delta_none_when_value_missing():
    a = _doc("a", models=[_model("m", wer=0.10)])
    b = _doc("b", models=[_model("other", wer=0.08)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    m = [x for x in rep["models"] if x["model_id"] == "m"][0]
    assert m["deltas"]["wer"] is None


def test_warning_on_corpus_mismatch():
    a = _doc("a", corpus="corpus-A", models=[_model("m")])
    b = _doc("b", corpus="corpus-B", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert any("corpus differs" in w for w in rep["warnings"])


def test_warning_on_beam_size_mismatch():
    a = _doc("a", models=[_model("m")],
             config={"device": "cuda", "compute_type": "float16",
                     "beam_size": 5, "vad_filter": True, "batch_size": 1})
    b = _doc("b", models=[_model("m")],
             config={"device": "cuda", "compute_type": "float16",
                     "beam_size": 1, "vad_filter": True, "batch_size": 1})
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert any("beam_size differs" in w for w in rep["warnings"])


def test_no_warnings_when_runs_match():
    a = _doc("a", models=[_model("m")])
    b = _doc("b", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert rep["warnings"] == []


def test_compare_runs_delta_requires_two_docs():
    a = _doc("a", models=[_model("m")])
    with pytest.raises(ValueError):
        asr_compare.compare_runs([a], mode="delta")


def test_fmt_pct_and_rtfx_and_none():
    assert asr_compare._fmt("wer", 0.089) == "8.9"
    assert asr_compare._fmt("rtfx", 64.8) == "64.8"
    assert asr_compare._fmt("wer", None) == "—"


def test_delta_mark_direction():
    assert asr_compare._delta_mark("wer", -0.02) == "✓"
    assert asr_compare._delta_mark("wer", 0.02) == "✗"
    assert asr_compare._delta_mark("rtfx", 10.0) == "✓"
    assert asr_compare._delta_mark("rtfx", -10.0) == "✗"
    assert asr_compare._delta_mark("wer", 0.0) == ""


def test_render_delta_has_models_metrics_and_marks():
    a = _doc("runA", models=[_model("big", "Big Model", wer=0.10, rtfx=60.0)])
    b = _doc("runB", models=[_model("big", "Big Model", wer=0.08, rtfx=70.0)])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    md = asr_compare.render_comparison_markdown(rep)
    assert "# ASR Run Comparison" in md
    assert "Big Model" in md
    assert "WER%" in md and "RTFx" in md
    assert "✓" in md
    assert "8.0" in md


def test_render_matrix_has_one_column_per_run():
    a = _doc("runA", models=[_model("m", "M", wer=0.10)])
    b = _doc("runB", models=[_model("m", "M", wer=0.09)])
    c = _doc("runC", models=[_model("m", "M", wer=0.08)])
    rep = asr_compare.compare_runs([a, b, c], mode="matrix")
    md = asr_compare.render_comparison_markdown(rep)
    assert "`runA`" in md and "`runB`" in md and "`runC`" in md
    assert "10.0" in md and "9.0" in md and "8.0" in md


def test_render_warnings_as_blockquotes():
    a = _doc("a", corpus="A", models=[_model("m")])
    b = _doc("b", corpus="B", models=[_model("m")])
    md = asr_compare.render_comparison_markdown(
        asr_compare.compare_runs([a, b], mode="delta"))
    assert "> ⚠️" in md
    assert "corpus differs" in md


def test_render_added_removed_show_dash():
    a = _doc("a", models=[_model("old", "Old", wer=0.10)])
    b = _doc("b", models=[_model("new", "New", wer=0.08)])
    md = asr_compare.render_comparison_markdown(
        asr_compare.compare_runs([a, b], mode="delta"))
    assert "removed" in md and "added" in md
    assert "—" in md


def test_per_clip_joins_by_audio_basename():
    ca = [{"audio": "wk1.mp4", "wer": 0.10, "der": None, "num_speakers": None}]
    cb = [{"audio": "wk1.mp4", "wer": 0.08, "der": None, "num_speakers": None}]
    a = _doc("a", models=[_model("m", "M", clips=ca)])
    b = _doc("b", models=[_model("m", "M", clips=cb)])
    rep = asr_compare.compare_runs([a, b], mode="delta", per_clip=True)
    assert rep["per_clip"] is True
    mrow = rep["models"][0]
    assert mrow["clip_order"] == ["wk1.mp4"]
    clip = mrow["clips"]["wk1.mp4"]
    assert clip["values"]["wer"] == [0.10, 0.08]
    assert round(clip["deltas"]["wer"], 4) == -0.02


def test_per_clip_render_has_clip_section():
    ca = [{"audio": "wk1.mp4", "wer": 0.10, "der": None, "num_speakers": None}]
    cb = [{"audio": "wk1.mp4", "wer": 0.08, "der": None, "num_speakers": None}]
    a = _doc("a", models=[_model("m", "Model M", clips=ca)])
    b = _doc("b", models=[_model("m", "Model M", clips=cb)])
    rep = asr_compare.compare_runs([a, b], mode="delta", per_clip=True)
    md = asr_compare.render_comparison_markdown(rep)
    assert "Per-clip: Model M" in md
    assert "wk1.mp4" in md


def test_per_clip_default_off():
    a = _doc("a", models=[_model("m")])
    b = _doc("b", models=[_model("m")])
    rep = asr_compare.compare_runs([a, b], mode="delta")
    assert rep["per_clip"] is False
    assert "clips" not in rep["models"][0]
