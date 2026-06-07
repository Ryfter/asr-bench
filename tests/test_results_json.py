import json
import types
from pathlib import Path
import pytest
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
    corpus = Path("/corpus")
    cmd = asr_bench._reproducibility_command(_cmd_args(), corpus, [_whisper_result()])
    assert cmd.startswith(f"python asr_bench.py --corpus '{corpus}' --models small,large-v3-turbo")
    assert "--device cuda" in cmd and "--compute-type float16" in cmd
    assert "--batch-size" not in cmd and "--beam-size" not in cmd and "--no-vad-filter" not in cmd


def test_reproducibility_command_nondefault_flags_and_nim():
    args = _cmd_args()
    args.batch_size = 8; args.beam_size = 3; args.vad_filter = False
    args.nim_model = "canary"
    cmd = asr_bench._reproducibility_command(args, Path("/c"), [_nim_result()])
    assert "--batch-size 8" in cmd and "--beam-size 3" in cmd and "--no-vad-filter" in cmd
    assert "--nim-url localhost:50051" in cmd and "--nim-model canary" in cmd
    assert "--nim-language en-US" in cmd


def _wx_result_with_nan():
    """A whisperx ModelResult whose clip has der=NaN to exercise sanitization."""
    clip = asr_bench.ClipResult(
        audio="lec.mp4", audio_sec=600.0, transcribe_sec=20.0, rtfx=30.0,
        vram_peak_bytes=None, hypothesis="hello world",
        reference_normalized="hello world", hypothesis_normalized="hello world",
        wer=0.10, mer=0.09, wil=0.12, hits=90, substitutions=5, deletions=3,
        insertions=2, cue_count=40, num_speakers=2, der=float("nan"),
        speaker_segments=[(0.0, 300.0, "SPEAKER_00"), (300.0, 600.0, "SPEAKER_01")],
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="large-v3-turbo+whisperx", display="Whisper Large V3 Turbo + WhisperX",
        fw_name="large-v3-turbo", params="809M", developer="OpenAI", languages="99",
        notes="x", disk_bytes=None, load_sec=0.0, engine="whisperx",
        vram_is_total=False, clips=[clip])


def _doc_args(**over):
    base = dict(models=["large-v3-turbo+whisperx"], device="cuda", compute_type="float16",
                batch_size=1, beam_size=5, vad_filter=True, nim_url="localhost:50051",
                nim_model="", nim_language="en-US", fuse=False, profile="both")
    base.update(over)
    return types.SimpleNamespace(**base)


def _doc_cfg():
    return asr_bench.RunConfig(
        device="cuda", compute_type="float16", diarize=True,
        hf_token="hf_SECRET", nim_api_key="nim_SECRET",
        min_speakers=2, max_speakers=2)


def test_build_document_top_level_shape():
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/corpus"), cfg=_doc_cfg(),
        args=_doc_args(), gold_label="**proxy** (default: pass --gold ...)",
        pairs=[], report_path=Path("report/20260605-120000.md"),
        generated_at="2026-06-05T12:00:00-06:00")
    assert doc["schema_version"] == 1
    assert doc["generated_at"] == "2026-06-05T12:00:00-06:00"
    assert doc["report_markdown"].endswith("20260605-120000.md")
    assert doc["command"].startswith("python asr_bench.py")
    assert doc["run"]["device"] == "cuda"
    assert doc["run"]["reference_quality"] == "proxy"
    assert doc["run"]["clips_count"] == 1
    assert len(doc["models"]) == 1


def test_build_document_redacts_secrets():
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(), args=_doc_args(),
        gold_label="proxy", pairs=[], report_path=Path("r.md"),
        generated_at="t")
    cfg_out = doc["run"]["config"]
    assert "hf_token" not in cfg_out and "nim_api_key" not in cfg_out
    assert cfg_out["diarize"] is True and cfg_out["min_speakers"] == 2
    import json as _json
    blob = _json.dumps(doc)
    assert "hf_SECRET" not in blob and "nim_SECRET" not in blob


def test_build_document_aggregates_and_clip_fields():
    m = _wx_result_with_nan()
    doc = asr_bench.build_results_document(
        [m], corpus=Path("/c"), cfg=_doc_cfg(), args=_doc_args(),
        gold_label="gold", pairs=[], report_path=Path("r.md"), generated_at="t")
    agg = doc["models"][0]["aggregates"]
    assert abs(agg["avg_wer"] - m.avg_wer) < 1e-9
    assert abs(agg["aggregate_rtfx"] - m.aggregate_rtfx) < 1e-9
    assert agg["peak_vram_bytes"] is None
    clip = doc["models"][0]["clips"][0]
    assert clip["der"] is None  # NaN -> null
    assert clip["num_speakers"] == 2
    assert clip["speaker_segments"][0] == {"start": 0.0, "end": 300.0, "speaker": "SPEAKER_00"}
    assert clip["hypothesis"] == "hello world"


def test_build_document_reference_quality_gold():
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(), args=_doc_args(),
        gold_label="**gold (hand-corrected, declared via --gold)**", pairs=[],
        report_path=Path("r.md"), generated_at="t")
    assert doc["run"]["reference_quality"] == "gold"


def test_build_document_fusion_stub_absent_and_present():
    off = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(),
        args=_doc_args(fuse=False), gold_label="proxy", pairs=[],
        report_path=Path("r.md"), generated_at="t")
    assert off["fusion"] == {"ran": False}
    on = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(),
        args=_doc_args(fuse=True, profile="verbatim"), pairs=[],
        gold_label="proxy", report_path=Path("r.md"), generated_at="t")
    assert on["fusion"]["ran"] is True
    assert on["fusion"]["profiles"] == ["verbatim"]


def test_build_document_fusion_outputs_listed():
    pair = asr_bench.Pair(audio=Path("/aud/Lec_default.mp4"), reference=Path("/aud/Lec.txt"))
    doc = asr_bench.build_results_document(
        [_wx_result_with_nan()], corpus=Path("/c"), cfg=_doc_cfg(),
        args=_doc_args(fuse=True, profile="both"), gold_label="proxy", pairs=[pair],
        report_path=Path("r.md"), generated_at="t")
    outs = doc["fusion"]["outputs"]
    assert any(o.endswith("Lec_Captions_Fused.vtt") for o in outs)
    assert any(o.endswith("Lec_KB_Fused.jsonl") for o in outs)
    assert any(o.endswith("Lec_KB_Fused.md") for o in outs)


def test_write_results_json_roundtrips(tmp_path):
    doc = {"schema_version": 1, "run": {"device": "cuda"}, "models": []}
    out = asr_bench.write_results_json(doc, tmp_path / "r.json")
    assert out == tmp_path / "r.json"
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1 and loaded["run"]["device"] == "cuda"


def test_write_results_json_no_nan_token(tmp_path):
    # A sanitized document never contains NaN; the written text must be valid JSON.
    doc = asr_bench._json_sanitize({"der": float("nan"), "wer": 0.1})
    out = asr_bench.write_results_json(doc, tmp_path / "r.json")
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert json.loads(text)["der"] is None


def test_sidecar_clip_has_cer():
    import asr_bench
    clip = asr_bench.ClipResult(
        audio="c.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, cer=0.05)
    d = asr_bench._clip_to_dict(clip)
    assert d["cer"] == 0.05


def test_sidecar_model_aggregates_have_new_speed_and_cer():
    import asr_bench
    clip = asr_bench.ClipResult(
        audio="c.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, cer=0.05)
    m = asr_bench.ModelResult(
        model_id="m", display="M", fw_name="m", params="1", developer="x",
        languages="en", notes="", disk_bytes=None, load_sec=0.0, clips=[clip])
    agg = asr_bench._model_to_dict(m)["aggregates"]
    assert agg["avg_cer"] == 0.05
    assert agg["median_rtfx"] == 60.0
    assert abs(agg["median_sec_per_audio_min"] - 1.0) < 1e-9


def test_write_results_json_raises_on_stray_nan(tmp_path):
    # Belt-and-suspenders: an unsanitized NaN must fail loudly, not emit invalid JSON.
    with pytest.raises(ValueError):
        asr_bench.write_results_json({"der": float("nan")}, tmp_path / "r.json")


def _fake_whisper_run(monkeypatch):
    """Patch a fake whisperx adapter + known audio duration so main() runs torch-free."""
    canned = asr_bench.WhisperXResult.from_dict(
        {"segments": [{"start": 0, "end": 2, "text": "hello world", "speaker": "SPEAKER_00"}],
         "speakers": ["SPEAKER_00"], "der": None, "language": "en"})
    monkeypatch.setattr(asr_bench, "make_whisperx_adapter",
                        lambda cfg: asr_bench.FakeWhisperXAdapter(canned))
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 2.0)


def test_main_writes_json_sidecar_with_output(tmp_path, monkeypatch):
    _fake_whisper_run(monkeypatch)
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    (tmp_path / "Lec.txt").write_text("hello world", encoding="utf-8")
    md = tmp_path / "out" / "report.md"
    monkeypatch.setattr("sys.argv", [
        "asr_bench.py", "--corpus", str(tmp_path), "--models", "small+whisperx",
        "--device", "cpu", "--no-diarize", "--output", str(md)])
    assert asr_bench.main() == 0
    js = tmp_path / "out" / "report.json"
    assert js.is_file()
    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["models"][0]["model_id"] == "small+whisperx"


def test_main_no_json_flag_skips_sidecar(tmp_path, monkeypatch):
    _fake_whisper_run(monkeypatch)
    audio = tmp_path / "Lec.mp4"; audio.write_bytes(b"x")
    (tmp_path / "Lec.txt").write_text("hello world", encoding="utf-8")
    md = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", [
        "asr_bench.py", "--corpus", str(tmp_path), "--models", "small+whisperx",
        "--device", "cpu", "--no-diarize", "--output", str(md), "--no-json"])
    assert asr_bench.main() == 0
    assert not (tmp_path / "report.json").exists()
