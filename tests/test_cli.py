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


def test_fuse_bad_llm_fails_fast(tmp_path, monkeypatch, capsys):
    import asr_bench
    # Fusion pre-flight runs before pairs discovery, so a bad --llm is caught
    # even on an empty corpus — without running the benchmark.
    monkeypatch.setattr("sys.argv", ["asr_bench.py", "--corpus", str(tmp_path), "--fuse", "--llm", "bogus:x"])
    rc = asr_bench.main()
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown --llm backend" in err or "bogus" in err


def test_init_context_writes_and_exits(tmp_path, monkeypatch):
    import asr_bench
    dest = tmp_path / "context.md"
    monkeypatch.setattr("sys.argv", ["asr_bench.py", "--init-context", str(dest)])
    rc = asr_bench.main()
    assert rc == 0
    assert dest.is_file()
    assert "Glossary" in dest.read_text(encoding="utf-8")


def test_help_lists_json_flag():
    import subprocess, sys
    out = subprocess.run([sys.executable, "asr_bench.py", "--help"],
                         capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0
    assert "--no-json" in out.stdout
