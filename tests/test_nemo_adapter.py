# tests/test_nemo_adapter.py
import json
import pytest
import asr_bench
from asr_bench import RunConfig, NeMoResult


def test_fake_adapter_returns_canned():
    canned = NeMoResult.from_dict({"text": "hi"})
    a = asr_bench.FakeNeMoAdapter(canned)
    assert a.name == "fake"
    assert a.transcribe("x.wav", "nvidia/parakeet-tdt-0.6b-v2",
                        RunConfig(device="cpu", compute_type="int8")) is canned


def test_runner_args_minimal():
    cfg = RunConfig(device="cuda", compute_type="float16")
    args = asr_bench._nemo_runner_args("a.wav", "nvidia/canary-qwen-2.5b", cfg)
    assert args[0] == asr_bench._NEMO_RUNNER_PATH
    assert "--audio" in args and "a.wav" in args
    assert "--model" in args and "nvidia/canary-qwen-2.5b" in args
    assert "--device" in args and "cuda" in args


def test_default_nemo_python_none_when_absent(tmp_path, monkeypatch):
    # Point module dir at an empty tmp dir → no .venv-nemo present.
    monkeypatch.setattr(asr_bench, "__file__", str(tmp_path / "asr_bench.py"))
    assert asr_bench._default_nemo_python() is None


def test_default_nemo_python_finds_venv(tmp_path, monkeypatch):
    venv = tmp_path / ".venv-nemo" / "Scripts"
    venv.mkdir(parents=True)
    (venv / "python.exe").write_text("", encoding="utf-8")
    monkeypatch.setattr(asr_bench, "__file__", str(tmp_path / "asr_bench.py"))
    got = asr_bench._default_nemo_python()
    assert got is not None and got.endswith("python.exe")


def test_make_nemo_adapter_errors_without_venv(monkeypatch):
    monkeypatch.setattr(asr_bench, "_default_nemo_python", lambda: None)
    cfg = RunConfig(device="cpu", compute_type="int8", nemo_python=None)
    with pytest.raises(RuntimeError, match="setup_nemo_venv"):
        asr_bench.make_nemo_adapter(cfg)


def test_make_nemo_adapter_uses_subprocess_with_venv(monkeypatch):
    monkeypatch.setattr(asr_bench, "_default_nemo_python", lambda: "/x/.venv-nemo/bin/python")
    cfg = RunConfig(device="cpu", compute_type="int8")
    a = asr_bench.make_nemo_adapter(cfg)
    assert isinstance(a, asr_bench.SubprocessNeMo)
    assert a.python == "/x/.venv-nemo/bin/python"


def test_subprocess_transcribe_parses_json_and_sets_env(monkeypatch):
    import subprocess
    from asr_bench import SubprocessNeMo, RunConfig
    captured = {}

    class _Proc:
        returncode = 0
        stdout = '{"text": "hello", "transcribe_sec": 1.0}'
        stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["env"] = kw.get("env")
        return _Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    adapter = SubprocessNeMo("python")
    cfg = RunConfig(device="cuda", compute_type="float16")
    result = adapter.transcribe("a.wav", "nvidia/canary-qwen-2.5b", cfg)

    assert result.text() == "hello"
    assert result.transcribe_sec == 1.0
    assert captured["env"]["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert "--model" in captured["cmd"] and "nvidia/canary-qwen-2.5b" in captured["cmd"]


def test_subprocess_transcribe_nonzero_raises(monkeypatch):
    import subprocess
    from asr_bench import SubprocessNeMo, RunConfig

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "boom traceback"

    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: _Proc())
    import pytest
    with pytest.raises(RuntimeError, match="boom traceback"):
        SubprocessNeMo("python").transcribe("a.wav", "m", RunConfig(device="cpu", compute_type="int8"))


def test_subprocess_transcribe_timeout_raises_runtimeerror(monkeypatch):
    import subprocess
    from asr_bench import SubprocessNeMo, RunConfig

    def fake_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", fake_run)
    import pytest
    with pytest.raises(RuntimeError, match="timed out"):
        SubprocessNeMo("python", timeout=5).transcribe(
            "a.wav", "m", RunConfig(device="cpu", compute_type="int8"))
