import types
from pathlib import Path
import asr_bench


def _whisper_result():
    clip = asr_bench.ClipResult(
        audio="lecture.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=200 * 1024**2, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.10, cue_count=50,
        reference_origin="unknown", reference_label="user-provided reference",
        mer=0.09, wil=0.12, hits=90, substitutions=5, deletions=3, insertions=2,
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
        mer=0.09, wil=0.12, hits=90, substitutions=5, deletions=3, insertions=2,
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
        show_alignment=False,
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


def test_whisper_row_has_no_star_marker():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    wl = [l for l in md.splitlines() if l.startswith("| Whisper Large V3 Turbo")][0]
    assert "*" not in wl


def test_headline_has_mer_and_wil_columns():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    header = [l for l in md.splitlines() if l.startswith("| Model | Params")][0]
    assert "MER%" in header and "WIL%" in header


def test_per_clip_table_has_sdi_columns():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "| S | D | I |" in md


def test_nan_metric_renders_as_dash():
    r = _whisper_result()
    r.clips[0].wer = float("nan")
    r.clips[0].mer = float("nan")
    r.clips[0].wil = float("nan")
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "nan" not in md.lower().split("reproducibility")[0]  # no literal 'nan' in the tables


def test_show_alignment_section_emitted_when_flag_on():
    args = _args()
    args.show_alignment = True
    r = _whisper_result()
    r.clips[0].reference_normalized = "the quick brown fox"
    r.clips[0].hypothesis_normalized = "the quick brown box"
    md = asr_bench.render_markdown([r], Path("."), args, "proxy")
    assert "## Alignment detail" in md


def test_no_alignment_section_by_default():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "## Alignment detail" not in md


def test_no_empty_alignment_header_when_all_clips_skipped():
    args = _args()
    args.show_alignment = True
    r = _whisper_result()
    r.clips[0].reference_normalized = ""
    r.clips[0].hypothesis_normalized = ""
    md = asr_bench.render_markdown([r], Path("."), args, "proxy")
    assert "## Alignment detail" not in md


def test_whisperx_disk_and_vram_render_na():
    clip = asr_bench.ClipResult(
        audio="lec.mp4", audio_sec=600.0, transcribe_sec=20.0, rtfx=30.0,
        vram_peak_bytes=None, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.10, mer=0.09, wil=0.12,
        num_speakers=1, der=float("nan"),
        reference_origin="unknown", reference_label="user-provided reference",
    )
    mr = asr_bench.ModelResult(
        model_id="small+whisperx", display="Whisper Small + WhisperX", fw_name="small",
        params="244M", developer="OpenAI", languages="99", notes="x", disk_bytes=None,
        load_sec=0.0, engine="whisperx", vram_is_total=False, clips=[clip])
    md = asr_bench.render_markdown([mr], Path("."), _args(), "proxy")
    row = [l for l in md.splitlines() if l.startswith("| Whisper Small + WhisperX")][0]
    assert "n/a" in row   # disk n/a; vram n/a
