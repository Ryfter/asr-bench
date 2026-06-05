import json
import sys
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


def test_parse_rttm_skips_comments_and_headers(tmp_path):
    content = (
        ";; this is a comment\n"
        "SPKR-INFO file 1 <NA> <NA> <NA> unknown A <NA> <NA>\n"
        "SPEAKER file 1 0.000 1.500 <NA> <NA> A <NA> <NA>\n"
        "\n"
        "SPEAKER file 1 1.500 0.500 <NA> <NA> B <NA> <NA>\n"
    )
    p = tmp_path / "f.rttm"; p.write_text(content, encoding="utf-8")
    segs = wr.parse_rttm(str(p))
    assert segs == [(0.0, 1.5, "A"), (1.5, 2.0, "B")]


def test_main_emits_only_json_on_stdout(monkeypatch, capsys):
    """Regression for the Task-13 live fix: whisperx/torch/pyannote write progress
    + logging to stdout, but the runner's contract is a single JSON document on
    stdout. main() must redirect that chatter to stderr so SubprocessWhisperX's
    json.loads(proc.stdout) never chokes on a leading log line."""
    def noisy_run(**kwargs):
        print("INFO: loading model / Performing voice activity detection...")
        sys.stdout.write('Downloading: "https://example/model.pth"\n')
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
                "words": [], "speakers": [], "der": None, "language": "en"}

    monkeypatch.setattr(wr, "run_whisperx", noisy_run)
    monkeypatch.setattr(sys, "argv",
                        ["whisperx_runner.py", "--audio", "a.wav", "--model", "small", "--device", "cpu"])
    rc = wr.main()
    assert rc == 0
    out, err = capsys.readouterr()
    parsed = json.loads(out)  # stdout MUST be pure JSON (no leading log lines)
    assert parsed["language"] == "en" and parsed["segments"][0]["text"] == "hi"
    assert "Downloading" not in out and "INFO" not in out
