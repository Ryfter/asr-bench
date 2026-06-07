import math
import asr_bench


# Morris/Maier/Green Table 1. Tokens chosen so each row reproduces the paper's
# H,S,D,I and integer %WER/%MER/%WIL. (a,b,c are distinct words.)
TABLE_1 = [
    # ref,            hyp,                  WER%, MER%, WIL%
    ("a",             "a",                  0,    0,    0),
    ("a",             "a a b b",            300,  75,   75),
    ("a b a",         "a c",                67,   67,   83),
    ("a",             "b",                  100,  100,  100),
    ("a",             "b c",                200,  100,  100),
]


def test_table1_wer_mer_wil_match_paper():
    for ref, hyp, wer_pct, mer_pct, wil_pct in TABLE_1:
        m = asr_bench.compute_word_metrics(ref, hyp)
        assert round(m.wer * 100) == wer_pct, (ref, hyp, "wer", m.wer)
        assert round(m.mer * 100) == mer_pct, (ref, hyp, "mer", m.mer)
        assert round(m.wil * 100) == wil_pct, (ref, hyp, "wil", m.wil)


def test_table1_hsdi_counts():
    # Row 3: ref "a b a" vs hyp "a c"  -> H=1, S=1, D=1, I=0
    m = asr_bench.compute_word_metrics("a b a", "a c")
    assert (m.hits, m.substitutions, m.deletions, m.insertions) == (1, 1, 1, 0)


def test_empty_reference_is_nan_not_crash():
    m = asr_bench.compute_word_metrics("", "")
    assert math.isnan(m.wer) and math.isnan(m.mer) and math.isnan(m.wil)


def test_empty_hypothesis_is_all_deletions():
    # non-empty ref, empty hyp -> H=0, S=0, D=3, I=0; WER/MER/WIL all 100%
    m = asr_bench.compute_word_metrics("a b c", "")
    assert (m.hits, m.substitutions, m.deletions, m.insertions) == (0, 0, 3, 0)
    assert round(m.wer * 100) == 100
    assert round(m.mer * 100) == 100
    assert round(m.wil * 100) == 100


def test_clipresult_has_metric_fields_with_defaults():
    # New fields must have defaults so existing positional constructors keep working.
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=1.0, transcribe_sec=1.0, rtfx=1.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="r",
        hypothesis_normalized="h", wer=0.5,
    )
    assert c.mer != c.mer or c.mer == 0 or isinstance(c.mer, float)  # field exists
    assert hasattr(c, "wil") and hasattr(c, "substitutions")


def test_compute_word_metrics_has_cer():
    m = asr_bench.compute_word_metrics("the cat", "the bat")
    # one character substitution ('c'->'b') over 7 reference chars
    assert abs(m.cer - 1.0 / 7.0) < 1e-6


def test_compute_word_metrics_empty_ref_cer_is_nan():
    m = asr_bench.compute_word_metrics("", "anything")
    assert math.isnan(m.cer)


def test_compute_word_metrics_perfect_match_cer_zero():
    m = asr_bench.compute_word_metrics("hello world", "hello world")
    assert m.cer == 0.0


def test_modelresult_avg_mer_wil():
    c1 = asr_bench.ClipResult(
        audio="a", audio_sec=1, transcribe_sec=1, rtfx=1, vram_peak_bytes=None,
        hypothesis="", reference_normalized="", hypothesis_normalized="",
        wer=0.2, mer=0.1, wil=0.3,
    )
    c2 = asr_bench.ClipResult(
        audio="b", audio_sec=1, transcribe_sec=1, rtfx=1, vram_peak_bytes=None,
        hypothesis="", reference_normalized="", hypothesis_normalized="",
        wer=0.4, mer=0.3, wil=0.5,
    )
    mr = asr_bench.ModelResult(
        model_id="m", display="M", fw_name="m", params="1", developer="d",
        languages="en", notes="", disk_bytes=None, load_sec=0.0, clips=[c1, c2],
    )
    assert abs(mr.avg_mer - 0.2) < 1e-9
    assert abs(mr.avg_wil - 0.4) < 1e-9


def test_clipresult_has_cer_field_default_nan():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1,
    )
    assert math.isnan(c.cer)  # default
    c2 = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, cer=0.05,
    )
    assert c2.cer == 0.05


# ---------------------------------------------------------------------------
# Task 3 helpers — avg_cer, median_rtfx, median_sec_per_audio_min
# ---------------------------------------------------------------------------

def _clip(rtfx=10.0, audio_sec=600.0, transcribe_sec=60.0, cer=0.10):
    return asr_bench.ClipResult(
        audio="c.mp4", audio_sec=audio_sec, transcribe_sec=transcribe_sec,
        rtfx=rtfx, vram_peak_bytes=None, hypothesis="h",
        reference_normalized="h", hypothesis_normalized="h", wer=0.1, cer=cer,
    )


def _model(clips):
    return asr_bench.ModelResult(
        model_id="m", display="M", fw_name="m", params="1", developer="x",
        languages="en", notes="", disk_bytes=None, load_sec=0.0, clips=clips,
    )


def test_avg_cer():
    m = _model([_clip(cer=0.10), _clip(cer=0.20)])
    assert abs(m.avg_cer - 0.15) < 1e-9


def test_median_rtfx_resists_outlier():
    clips = [_clip(rtfx=60.0, audio_sec=600.0, transcribe_sec=10.0),
             _clip(rtfx=62.0, audio_sec=600.0, transcribe_sec=9.7),
             _clip(rtfx=3.0, audio_sec=600.0, transcribe_sec=200.0)]
    m = _model(clips)
    assert m.median_rtfx == 60.0
    assert m.median_rtfx > m.aggregate_rtfx  # outlier resistance


def test_median_sec_per_audio_min():
    m = _model([_clip(audio_sec=600.0, transcribe_sec=10.0)])
    assert abs(m.median_sec_per_audio_min - 1.0) < 1e-9


def test_median_sec_per_audio_min_skips_zero_audio():
    m = _model([_clip(audio_sec=0.0, transcribe_sec=5.0),
                _clip(audio_sec=600.0, transcribe_sec=10.0)])
    assert abs(m.median_sec_per_audio_min - 1.0) < 1e-9


def test_median_properties_empty_model():
    m = _model([])
    assert m.avg_cer == 0.0
    assert m.median_rtfx == 0.0
    assert m.median_sec_per_audio_min == 0.0


# ---------------------------------------------------------------------------
# Task 1 — reference-free hallucination signals
# ---------------------------------------------------------------------------

def test_repeat_coverage_high_on_loop():
    text = "thank you so much " * 6  # 24 words, heavy 4-gram repetition
    cov = asr_bench._repeat_coverage(text.strip())
    assert cov > 0.30


def test_repeat_coverage_zero_on_clean_prose():
    text = "the quick brown fox jumps over the lazy dog near twelve silent owls"
    assert asr_bench._repeat_coverage(text) == 0.0


def test_repeat_coverage_short_text_guard():
    assert asr_bench._repeat_coverage("one two three") == 0.0  # < 8 words


def test_compression_ratio_high_on_repetition():
    text = "thank you so much for watching this video. " * 20  # > 200 chars, repetitive
    assert asr_bench._compression_ratio(text) > 2.4


def test_compression_ratio_short_text_guard():
    assert asr_bench._compression_ratio("short text") == 1.0  # < 200 chars


def test_compute_hallucination_signals_returns_pair():
    loop = "thank you so much " * 20
    cov, ratio = asr_bench.compute_hallucination_signals(loop, loop.strip())
    assert cov > 0.30
    assert ratio > 2.4


# ---------------------------------------------------------------------------
# Task 2 — ClipResult hallucination fields + is_hallucination_suspect
# ---------------------------------------------------------------------------

def test_clipresult_hallucination_fields_default():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1)
    assert c.repeat_coverage == 0.0
    assert c.compression_ratio == 1.0
    assert c.is_hallucination_suspect is False


def test_is_hallucination_suspect_on_repeat_coverage():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.5,
        compression_ratio=1.5)
    assert c.is_hallucination_suspect is True


def test_is_hallucination_suspect_on_compression():
    c = asr_bench.ClipResult(
        audio="x.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.0,
        compression_ratio=3.0)
    assert c.is_hallucination_suspect is True


# ---------------------------------------------------------------------------
# Task 3 — ModelResult.hallucination_rate
# ---------------------------------------------------------------------------

def test_hallucination_rate_half():
    clean = asr_bench.ClipResult(
        audio="a.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1)  # defaults -> not suspect
    suspect = asr_bench.ClipResult(
        audio="b.mp4", audio_sec=10.0, transcribe_sec=1.0, rtfx=10.0,
        vram_peak_bytes=None, hypothesis="h", reference_normalized="h",
        hypothesis_normalized="h", wer=0.1, repeat_coverage=0.6)
    m = _model([clean, suspect])
    assert m.hallucination_rate == 0.5


def test_hallucination_rate_empty_model():
    assert _model([]).hallucination_rate == 0.0
