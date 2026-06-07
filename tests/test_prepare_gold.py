"""B2 — `prepare-gold` subcommand: convert VTT/SRT caption files into the plain
.txt reference files asr-bench scores against. Pure (no torch)."""
from pathlib import Path

import asr_bench


_VTT = """WEBVTT

1
00:00:00.000 --> 00:00:02.000
Hello world

2
00:00:02.000 --> 00:00:04.000
this is a test
"""

_SRT = """1
00:00:00,000 --> 00:00:02,000
Hello world

2
00:00:02,000 --> 00:00:04,000
this is a test
"""

_PROXY_VTT = """WEBVTT

[Auto-generated transcript. Edits may have been applied for clarity.]

1
00:00:00.000 --> 00:00:02.000
machine made this
"""


def test_converts_vtt_to_flat_txt(tmp_path):
    src = tmp_path / "lecture.vtt"
    src.write_text(_VTT, encoding="utf-8")
    rc = asr_bench.prepare_gold_main([str(tmp_path)])
    assert rc == 0
    out = tmp_path / "lecture.txt"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Hello world this is a test" in text
    assert "00:00" not in text          # timing stripped
    assert "WEBVTT" not in text


def test_converts_srt_to_flat_txt(tmp_path):
    src = tmp_path / "lecture.srt"
    src.write_text(_SRT, encoding="utf-8")
    asr_bench.prepare_gold_main([str(tmp_path)])
    text = (tmp_path / "lecture.txt").read_text(encoding="utf-8")
    assert "Hello world this is a test" in text
    assert "-->" not in text


def test_skips_existing_txt_without_overwrite(tmp_path):
    (tmp_path / "lecture.vtt").write_text(_VTT, encoding="utf-8")
    out = tmp_path / "lecture.txt"
    out.write_text("ORIGINAL HUMAN GOLD", encoding="utf-8")
    asr_bench.prepare_gold_main([str(tmp_path)])
    assert out.read_text(encoding="utf-8") == "ORIGINAL HUMAN GOLD"  # untouched


def test_overwrite_replaces_existing_txt(tmp_path):
    (tmp_path / "lecture.vtt").write_text(_VTT, encoding="utf-8")
    out = tmp_path / "lecture.txt"
    out.write_text("STALE", encoding="utf-8")
    asr_bench.prepare_gold_main([str(tmp_path), "--overwrite"])
    assert "Hello world" in out.read_text(encoding="utf-8")


def test_excludes_generated_caption_vtts(tmp_path):
    # asr-bench's own output must never be converted into a reference (circular)
    gen = tmp_path / "lecture_Captions_Whisper Small.vtt"
    gen.write_text(_VTT, encoding="utf-8")
    rc = asr_bench.prepare_gold_main([str(tmp_path)])
    assert rc == 1                       # nothing convertible found
    assert not (tmp_path / "lecture_Captions_Whisper Small.txt").exists()


def test_dry_run_writes_nothing(tmp_path, capsys):
    (tmp_path / "lecture.vtt").write_text(_VTT, encoding="utf-8")
    asr_bench.prepare_gold_main([str(tmp_path), "--dry-run"])
    assert not (tmp_path / "lecture.txt").exists()
    assert "dry" in capsys.readouterr().out.lower()


def test_proxy_source_keeps_proxy_marker(tmp_path, capsys):
    src = tmp_path / "auto.vtt"
    src.write_text(_PROXY_VTT, encoding="utf-8")
    asr_bench.prepare_gold_main([str(tmp_path)])
    out = tmp_path / "auto.txt"
    body = out.read_text(encoding="utf-8")
    # the auto-generated marker survives so the .txt is still detected as proxy
    assert "[Auto-generated transcript" in body
    origin, _ = asr_bench.detect_reference_origin(out)
    assert origin != "unknown"           # still flagged proxy, not laundered to gold
    assert "PROXY" in capsys.readouterr().out
    # ...but the scored text is clean (the bracketed line is stripped at scoring)
    assert asr_bench.load_reference_text(out).strip() == "machine made this"


def test_returns_1_when_nothing_to_convert(tmp_path, capsys):
    rc = asr_bench.prepare_gold_main([str(tmp_path)])
    assert rc == 1
    assert "No .vtt/.srt" in capsys.readouterr().err
