import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_help_lists_nim_flags():
    out = subprocess.run(
        [sys.executable, "asr_bench.py", "--help"],
        capture_output=True, text=True, cwd=str(ROOT),
    ).stdout
    assert "--nim-url" in out
    assert "--nim-model" in out
    assert "--nim-language" in out
    assert "--nim-api-key" in out


def test_adhoc_nim_id_accepted_not_rejected_as_unknown(tmp_path):
    # Empty (but existing) corpus dir: model validation runs and must ACCEPT nim:foo,
    # then the run exits on "no pairs". If nim:foo were rejected we'd see "unknown models".
    res = subprocess.run(
        [sys.executable, "asr_bench.py", "--models", "nim:foo", "--corpus", str(tmp_path)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    combined = (res.stdout + res.stderr).lower()
    assert "unknown models" not in combined
    assert "no (audio, reference) pairs" in combined  # confirms validation passed, reached discovery
