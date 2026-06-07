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
