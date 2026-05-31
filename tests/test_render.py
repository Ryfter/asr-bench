import types
from pathlib import Path
import asr_bench


def _whisper_result():
    clip = asr_bench.ClipResult(
        audio="lecture.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=200 * 1024**2, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.10, cue_count=50,
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="large-v3-turbo", display="Whisper Large V3 Turbo",
        fw_name="large-v3-turbo", params="809M", developer="OpenAI",
        languages="99", notes="x", disk_bytes=1600 * 1024**2, load_sec=2.0,
        engine="faster-whisper", vram_is_total=False, clips=[clip],
    )


def _nim_result():
    clip = asr_bench.ClipResult(
        audio="lecture.mp4", audio_sec=600.0, transcribe_sec=8.0, rtfx=75.0,
        vram_peak_bytes=9 * 1024**3, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.09, cue_count=48,
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="canary-nim", display="Canary (NIM)", fw_name="", params="—",
        developer="NVIDIA", languages="en", notes="x", disk_bytes=None,
        load_sec=0.5, engine="nim", vram_is_total=True, clips=[clip],
    )


def _args():
    return types.SimpleNamespace(
        device="cuda", compute_type="float16", batch_size=1, beam_size=5,
        vad_filter=True, models=["large-v3-turbo", "canary-nim"],
        nim_url="localhost:50051", nim_model="", nim_language="en-US",
        nim_api_key=None, nim_ssl=False,
    )


def test_nim_disk_renders_na():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    nim_line = [l for l in md.splitlines() if l.startswith("| Canary (NIM)")][0]
    assert "n/a" in nim_line


def test_nim_vram_has_star_marker():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    nim_line = [l for l in md.splitlines() if l.startswith("| Canary (NIM)")][0]
    assert "*" in nim_line


def test_engines_note_present_when_nim_in_run():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    assert "Engines in this run" in md
    assert "total GPU memory" in md


def test_reproducibility_includes_nim_flags():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    assert "--nim-url localhost:50051" in md


def test_no_engines_note_for_whisper_only():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "Engines in this run" not in md
