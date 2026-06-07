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


def _whisperx_result():
    clip = asr_bench.ClipResult(
        audio="lec.mp4", audio_sec=600.0, transcribe_sec=20.0, rtfx=30.0,
        vram_peak_bytes=None, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.10, mer=0.09, wil=0.12,
        hits=90, substitutions=5, deletions=3, insertions=2,
        num_speakers=2, der=0.15,
        speaker_segments=[(0.0, 300.0, "SPEAKER_00"), (300.0, 600.0, "SPEAKER_01")],
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="large-v3-turbo+whisperx", display="Whisper Large V3 Turbo + WhisperX",
        fw_name="large-v3-turbo", params="809M", developer="OpenAI", languages="99",
        notes="x", disk_bytes=None, load_sec=0.0, engine="whisperx",
        vram_is_total=False, clips=[clip])


def test_der_and_speakers_shown_when_whisperx():
    md = asr_bench.render_markdown([_whisperx_result()], Path("."), _args(), "proxy")
    assert "DER%" in md and "Speakers" in md
    assert "15.0" in md   # der 0.15 -> 15.0


def test_der_absent_for_plain_whisper():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "DER%" not in md


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


def test_headline_has_cer_and_median_rtfx_columns():
    r = _whisper_result()
    r.clips[0].cer = 0.05
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    header = [l for l in md.splitlines() if l.startswith("| Model | Params")][0]
    assert "CER%" in header
    assert "RTFx (med)" in header


def test_headline_renders_median_rtfx_value():
    r = _whisper_result()
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "60.00x" in md   # _whisper_result clip rtfx=60.0 → median 60.00x


def test_per_clip_view_has_cer_column():
    r = _whisper_result()
    r.clips[0].cer = 0.05
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert any("| Model | WER% | MER% | WIL% | CER% |" in l for l in md.splitlines())


def test_per_model_breakdown_has_cer_column():
    r = _whisper_result()
    r.clips[0].cer = 0.05
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert any("| Clip | Audio | WER% | MER% | WIL% | CER% |" in l for l in md.splitlines())


# C2 — median_sec_per_audio_min surfaced in the headline table
def test_headline_has_median_sec_per_audio_min_column():
    r = _whisper_result()
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    header = [l for l in md.splitlines() if l.startswith("| Model | Params")][0]
    assert "s/aud-min (med)" in header


def test_headline_renders_median_sec_per_audio_min_value():
    r = _whisper_result()
    # clip audio_sec=600, transcribe_sec=10 → 10*60/600 = 1.00 s per audio-min
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "1.00s" in md


# C3 — pipe / newline escaping in free-text table cells
def test_headline_escapes_pipe_in_model_display():
    r = _whisper_result()
    r.display = "Weird|Model"
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    headline = [l for l in md.splitlines() if l.startswith("|") and "Weird" in l][0]
    assert "Weird\\|Model" in headline      # pipe escaped
    assert "Weird|Model" not in headline    # no raw pipe that would split the row


def test_per_model_breakdown_escapes_pipe_in_clip_name():
    r = _whisper_result()
    r.clips[0].audio = "lec|ture.mp4"
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    clip_row = [l for l in md.splitlines() if l.startswith("| lec")][0]
    assert "lec\\|ture.mp4" in clip_row


def test_cer_nan_renders_as_dash_not_nan():
    r = _whisper_result()
    r.clips[0].cer = float("nan")
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "nan" not in md.lower().split("reproducibility")[0]


def test_hallucination_section_appears_when_flagged():
    r = _whisper_result()
    r.clips[0].repeat_coverage = 0.6   # trips the flag
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "Hallucination signals" in md
    assert r.clips[0].audio in md
    assert "1/1 clip" in md or "1/1 clips" in md  # per-model summary


def test_no_hallucination_section_when_clean():
    r = _whisper_result()  # defaults -> not suspect
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "Hallucination signals" not in md


def test_hallucination_insertion_burst_note():
    r = _whisper_result()
    c = r.clips[0]
    c.compression_ratio = 3.0           # trips the flag (compression)
    c.hits, c.substitutions, c.deletions, c.insertions = 10, 0, 0, 20  # insertion rate 2.0
    md = asr_bench.render_markdown([r], Path("."), _args(), "proxy")
    assert "Hallucination signals" in md
    assert "insertion burst" in md.lower()
