import json
from pathlib import Path

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
