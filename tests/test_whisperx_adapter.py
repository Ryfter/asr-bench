import json
import asr_bench
from asr_bench import RunConfig, WhisperXResult


def test_fake_adapter_returns_result():
    canned = WhisperXResult.from_dict({"segments": [{"start": 0, "end": 1, "text": "hi"}], "language": "en"})
    a = asr_bench.FakeWhisperXAdapter(canned)
    out = a.transcribe("x.wav", model="small", cfg=RunConfig(device="cpu", compute_type="int8"), rttm=None)
    assert out.text() == "hi"


def test_make_adapter_prefers_inprocess_when_torch(monkeypatch):
    monkeypatch.setattr(asr_bench.importlib.util, "find_spec",
                        lambda name: object() if name == "torch" else None)
    a = asr_bench.make_whisperx_adapter(RunConfig(device="cpu", compute_type="int8"))
    assert isinstance(a, asr_bench.InProcessWhisperX)


def test_make_adapter_subprocess_when_no_torch_but_venv(monkeypatch, tmp_path):
    monkeypatch.setattr(asr_bench.importlib.util, "find_spec", lambda name: None)
    py = tmp_path / "python.exe"; py.write_bytes(b"x")
    a = asr_bench.make_whisperx_adapter(RunConfig(device="cpu", compute_type="int8",
                                                  whisperx_python=str(py)))
    assert isinstance(a, asr_bench.SubprocessWhisperX)
    assert a.python == str(py)


def test_make_adapter_errors_when_neither(monkeypatch):
    monkeypatch.setattr(asr_bench.importlib.util, "find_spec", lambda name: None)
    try:
        asr_bench.make_whisperx_adapter(RunConfig(device="cpu", compute_type="int8"))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "whisperx" in str(e).lower()


def test_subprocess_adapter_parses_json(monkeypatch, tmp_path):
    py = tmp_path / "python.exe"; py.write_bytes(b"x")
    payload = {"segments": [{"start": 0, "end": 1, "text": "sub out", "speaker": "SPEAKER_00"}],
               "speakers": ["SPEAKER_00"], "der": 0.2, "language": "en"}

    class FakeCompleted:
        stdout = json.dumps(payload)
        stderr = ""
        returncode = 0

    calls = {}
    def fake_run(cmd, capture_output=None, text=None, timeout=None, check=None, env=None):
        calls["cmd"] = cmd
        calls["env"] = env
        return FakeCompleted()

    monkeypatch.setattr(asr_bench.shutil, "which", lambda n: None)
    monkeypatch.setattr(asr_bench.subprocess, "run", fake_run)
    a = asr_bench.SubprocessWhisperX(str(py))
    cfg = RunConfig(device="cpu", compute_type="int8", diarize=True, hf_token="tok")
    out = a.transcribe(str(tmp_path / "a.wav"), model="small", cfg=cfg, rttm=str(tmp_path / "a.rttm"))
    assert out.der == 0.2 and out.speakers == ["SPEAKER_00"]
    assert any("whisperx_runner.py" in str(c) for c in calls["cmd"])
    assert "--rttm" in calls["cmd"]
    assert "--hf-token" not in calls["cmd"]          # token NOT in argv
    assert calls["env"].get("HF_TOKEN") == "tok"     # token passed via env
