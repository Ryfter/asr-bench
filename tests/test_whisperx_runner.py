import json
import pytest
import whisperx_runner as wr


RTTM = (
    "SPEAKER file 1 0.000 2.000 <NA> <NA> A <NA> <NA>\n"
    "SPEAKER file 1 2.000 2.000 <NA> <NA> B <NA> <NA>\n"
)


def test_parse_rttm(tmp_path):
    p = tmp_path / "file.rttm"; p.write_text(RTTM, encoding="utf-8")
    segs = wr.parse_rttm(str(p))
    assert segs == [(0.0, 2.0, "A"), (2.0, 4.0, "B")]


def test_build_arg_parser_defaults():
    ns = wr.build_arg_parser().parse_args(["--audio", "a.wav", "--model", "small", "--device", "cpu"])
    assert ns.audio == "a.wav" and ns.model == "small" and ns.diarize is False
    ns2 = wr.build_arg_parser().parse_args(
        ["--audio", "a.wav", "--model", "small", "--device", "cuda", "--diarize", "--rttm", "r.rttm"])
    assert ns2.diarize is True and ns2.rttm == "r.rttm"


def test_compute_der_perfect_match(tmp_path):
    pytest.importorskip("pyannote.metrics")
    p = tmp_path / "file.rttm"; p.write_text(RTTM, encoding="utf-8")
    hyp = [(0.0, 2.0, "A"), (2.0, 4.0, "B")]
    assert abs(wr.compute_der_from_rttm(hyp, str(p))) < 1e-9


def test_compute_der_all_wrong(tmp_path):
    pytest.importorskip("pyannote.metrics")
    p = tmp_path / "file.rttm"; p.write_text(RTTM, encoding="utf-8")
    hyp = [(0.0, 4.0, "X")]
    der = wr.compute_der_from_rttm(hyp, str(p))
    assert abs(der - 0.5) < 1e-6
