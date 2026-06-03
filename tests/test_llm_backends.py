import asr_bench


def test_fake_backend_returns_canned():
    b = asr_bench.FakeLLMBackend(lambda prompt: "FUSED:" + prompt[:3])
    assert b.generate("hello") == "FUSED:hel"


def test_make_llm_backend_fake():
    b = asr_bench.make_llm_backend("fake")
    assert isinstance(b, asr_bench.FakeLLMBackend)


def test_make_llm_backend_ollama_parses_model():
    b = asr_bench.make_llm_backend("ollama:qwen2.5")
    assert isinstance(b, asr_bench.OllamaBackend)
    assert b.model == "qwen2.5"


def test_make_llm_backend_cli_parses_command():
    b = asr_bench.make_llm_backend("cli:claude -p")
    assert isinstance(b, asr_bench.CliBackend)
    assert b.command == ["claude", "-p"]


def test_make_llm_backend_unknown_raises():
    try:
        asr_bench.make_llm_backend("nope:x")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cli_backend_invokes_subprocess(monkeypatch):
    calls = {}

    class FakeCompleted:
        stdout = "fused text"
        returncode = 0

    def fake_run(cmd, input=None, capture_output=None, text=None, timeout=None, check=None):
        calls["cmd"] = cmd
        calls["input"] = input
        return FakeCompleted()

    monkeypatch.setattr(asr_bench.subprocess, "run", fake_run)
    b = asr_bench.CliBackend(["claude", "-p"])
    out = b.generate("my prompt")
    assert out == "fused text"
    assert calls["cmd"] == ["claude", "-p"]
    assert calls["input"] == "my prompt"


def test_ollama_backend_posts_prompt(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout=None):
        import io, json
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))

        class Resp:
            def read(self_inner):
                return json.dumps({"response": "ollama fused"}).encode("utf-8")
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False
        return Resp()

    monkeypatch.setattr(asr_bench.urllib.request, "urlopen", fake_urlopen)
    b = asr_bench.OllamaBackend(model="qwen2.5")
    out = b.generate("hi")
    assert out == "ollama fused"
    assert captured["body"]["model"] == "qwen2.5"
    assert captured["body"]["prompt"] == "hi"
