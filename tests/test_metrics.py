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
