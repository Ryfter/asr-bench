import asr_bench


VTT = """WEBVTT

1
00:00:00.000 --> 00:00:02.500
Hello world

2
00:00:02.500 --> 00:00:05.000
this is a test
"""

SRT = """1
00:00:00,000 --> 00:00:02,500
Hello world

2
00:00:02,500 --> 00:00:05,000
this is a test
"""


def test_parse_vtt_cues(tmp_path):
    p = tmp_path / "cap.vtt"
    p.write_text(VTT, encoding="utf-8")
    cues = asr_bench.parse_caption_cues(p)
    assert len(cues) == 2
    assert cues[0].start == 0.0 and abs(cues[0].end - 2.5) < 1e-6
    assert cues[0].text == "Hello world"
    assert cues[1].text == "this is a test"


def test_parse_srt_cues(tmp_path):
    p = tmp_path / "cap.srt"
    p.write_text(SRT, encoding="utf-8")
    cues = asr_bench.parse_caption_cues(p)
    assert len(cues) == 2
    assert abs(cues[1].end - 5.0) < 1e-6


def test_parse_skips_panopto_header(tmp_path):
    p = tmp_path / "cap.vtt"
    p.write_text("WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.000\n[Auto-generated transcript.]\nreal text\n", encoding="utf-8")
    cues = asr_bench.parse_caption_cues(p)
    assert cues[0].text == "real text"
