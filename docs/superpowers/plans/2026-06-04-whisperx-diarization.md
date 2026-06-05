# asr-bench v0.3 — WhisperX + Diarization + DER — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a WhisperX engine (wav2vec2 word-alignment + pyannote speaker diarization) that produces speaker-labeled output and a DER metric, runnable either in-process (when torch is importable) or via a 3.12-venv subprocess bridge.

**Architecture:** A standalone `whisperx_runner.py` owns all torch/whisperx/pyannote imports and emits JSON; `asr_bench.py` gains a `WhisperXEngine` (Engine ABC) plus a `WhisperXAdapter` abstraction (InProcess / Subprocess / Fake, auto-selected) that produces a `WhisperXResult`, converted to a `ClipResult` (text → existing WER/MER/WIL; speaker segments + DER for the report). DER is computed in the runner (where pyannote lives) and gated on an `<audio-base>.rttm` sidecar.

**Tech Stack:** Python 3.14 (asr-bench core, no torch) + Python 3.12 venv (torch/whisperX/pyannote). jiwer (existing metrics), pyannote.metrics (DER, in runner), pytest. Tasks 1–12 need no torch; Task 13 is the live install + validation.

---

## File Structure

- **`whisperx_runner.py`** (create) — standalone. Heavy imports (`torch`, `whisperx`, `pyannote`) live ONLY inside functions. Public:
  - `run_whisperx(audio, model, device, language, diarize, hf_token, min_speakers, max_speakers, rttm) -> dict` — the shared core (used by both adapters).
  - `compute_der_from_rttm(hyp_segments, rttm_path) -> float` — pyannote.metrics DER (no torch).
  - `parse_rttm(path) -> list[tuple]` — manual RTTM → `[(start, end, speaker)]`.
  - `__main__` CLI: parses args, calls `run_whisperx`, prints JSON to stdout.
- **`asr_bench.py`** (modify):
  - `MODELS`/`resolve_model_entry`: `<size>+whisperx` resolution.
  - `RunConfig`: whisperx fields. `ClipResult`: speaker fields.
  - `WhisperXResult` dataclass + `WhisperXResult.from_dict`.
  - `WhisperXAdapter` ABC + `FakeWhisperXAdapter` + `InProcessWhisperX` + `SubprocessWhisperX` + `make_whisperx_adapter`.
  - `write_whisperx_vtt`, `write_words_sidecar`, `find_rttm`.
  - `WhisperXEngine(Engine)`; register in `ENGINES`.
  - `render_markdown`: DER% + Speakers columns (conditional). `main()`: CLI flags + cfg + pre-flight.
- **Tests:** `tests/test_whisperx_resolve.py`, `tests/test_whisperx_result.py`, `tests/test_whisperx_adapter.py`, `tests/test_whisperx_engine.py`, `tests/test_whisperx_vtt.py`, `tests/test_whisperx_runner.py`, plus additions to `tests/test_render.py` and `tests/test_cli.py`.
- **Docs:** `README.md`, `CLAUDE.md`, `SPEC.md`.

Tests import via `import asr_bench` / `import whisperx_runner` (repo root on `sys.path` via `tests/conftest.py`).

---

## Task 1: `WhisperXResult` dataclass + `from_dict`

**Files:**
- Modify: `asr_bench.py` — new `# ---- WhisperX ----` section after the `ENGINES = {...}` block (~line 1104)
- Test: `tests/test_whisperx_result.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import math
import asr_bench


def test_from_dict_full():
    d = {
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello", "speaker": "SPEAKER_00"}],
        "words": [{"word": "hello", "start": 0.0, "end": 0.5, "score": 0.9, "speaker": "SPEAKER_00"}],
        "speakers": ["SPEAKER_00"],
        "der": 0.12,
        "language": "en",
    }
    r = asr_bench.WhisperXResult.from_dict(d)
    assert r.segments[0]["text"] == "hello"
    assert r.speakers == ["SPEAKER_00"]
    assert r.der == 0.12
    assert r.language == "en"
    assert r.text() == "hello"


def test_from_dict_minimal_no_diarization():
    d = {"segments": [{"start": 0, "end": 1, "text": "a"}, {"start": 1, "end": 2, "text": "b"}],
         "language": "en"}
    r = asr_bench.WhisperXResult.from_dict(d)
    assert r.der is None
    assert r.speakers == []
    assert r.words == []
    assert r.text() == "a b"


def test_speaker_segments_helper():
    d = {"segments": [{"start": 0, "end": 1, "text": "a", "speaker": "SPEAKER_00"},
                      {"start": 1, "end": 2, "text": "b", "speaker": "SPEAKER_01"}],
         "speakers": ["SPEAKER_00", "SPEAKER_01"], "language": "en"}
    r = asr_bench.WhisperXResult.from_dict(d)
    assert r.speaker_segments() == [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_result.py -v` → FAIL (`WhisperXResult` missing).

- [ ] **Step 3: Implement** — add after `ENGINES = {...}`:

```python
# ---- WhisperX ---------------------------------------------------------------
@dataclass
class WhisperXResult:
    """Parsed output of a WhisperX run (transcribe + align + optional diarize)."""
    segments: List[Dict]                 # [{start, end, text, speaker?}]
    words: List[Dict] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    der: Optional[float] = None
    language: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "WhisperXResult":
        return cls(
            segments=list(d.get("segments") or []),
            words=list(d.get("words") or []),
            speakers=list(d.get("speakers") or []),
            der=d.get("der"),
            language=d.get("language") or "",
        )

    def text(self) -> str:
        return " ".join(s.get("text", "").strip() for s in self.segments).strip()

    def speaker_segments(self) -> List[Tuple[float, float, str]]:
        out: List[Tuple[float, float, str]] = []
        for s in self.segments:
            if s.get("speaker"):
                out.append((float(s["start"]), float(s["end"]), s["speaker"]))
        return out
```

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_result.py -v` → PASS (3). Then `python -m pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_result.py
git commit -m "feat: WhisperXResult dataclass + from_dict parsing"
```

---

## Task 2: `<size>+whisperx` model resolution

**Files:**
- Modify: `asr_bench.py` — `resolve_model_entry` (~159) + a module regex near `_NIM_ADHOC_RE` (~156)
- Test: `tests/test_whisperx_resolve.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import asr_bench


def test_resolve_whisperx_model():
    e = asr_bench.resolve_model_entry("large-v3-turbo+whisperx")
    assert e["engine"] == "whisperx"
    assert e["fw_name"] == "large-v3-turbo"
    assert e["id"] == "large-v3-turbo+whisperx"
    assert "WhisperX" in e["display"]


def test_resolve_whisperx_all_sizes():
    for size in ["small", "medium", "large-v3", "large-v3-turbo"]:
        e = asr_bench.resolve_model_entry(f"{size}+whisperx")
        assert e["engine"] == "whisperx" and e["fw_name"] == size


def test_resolve_whisperx_bad_size_errors():
    try:
        asr_bench.resolve_model_entry("bogus+whisperx")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_plain_size_still_faster_whisper():
    assert asr_bench.resolve_model_entry("small")["engine"] == "faster-whisper"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_resolve.py -v` → FAIL.

- [ ] **Step 3: Implement** — add the regex after `_NIM_ADHOC_RE` (~156):

```python
_WHISPERX_RE = re.compile(r"^(.+)\+whisperx$")
_WHISPERX_SIZES = {"small", "medium", "large-v3", "large-v3-turbo"}
```

In `resolve_model_entry`, after the `if model_id in MODELS:` block and before the `_NIM_ADHOC_RE` block, add:

```python
    wx = _WHISPERX_RE.match(model_id)
    if wx:
        size = wx.group(1).strip()
        if size not in _WHISPERX_SIZES:
            raise ValueError(
                f"unknown WhisperX base size '{size}' in '{model_id}' "
                f"(choices: {', '.join(sorted(_WHISPERX_SIZES))})"
            )
        base = MODELS[size]
        return {
            "id": model_id,
            "engine": "whisperx",
            "display": f"{base['display']} + WhisperX",
            "developer": base["developer"],
            "params": base["params"],
            "languages": base["languages"],
            "fw_name": size,
            "notes": "WhisperX: wav2vec2 word alignment + pyannote diarization.",
        }
```

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_resolve.py -v` → PASS (4). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_resolve.py
git commit -m "feat: resolve <size>+whisperx model ids to the whisperx engine"
```

---

## Task 3: `ClipResult` speaker fields + `RunConfig` whisperx fields

**Files:**
- Modify: `asr_bench.py` — `ClipResult` (~592, after `insertions`/before `cue_count`), `RunConfig` (~668)
- Test: `tests/test_whisperx_result.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_clipresult_speaker_fields_default():
    c = asr_bench.ClipResult(
        audio="a.mp4", audio_sec=1, transcribe_sec=1, rtfx=1, vram_peak_bytes=None,
        hypothesis="h", reference_normalized="r", hypothesis_normalized="h", wer=0.1,
    )
    assert c.speaker_segments == []
    assert c.num_speakers == 0
    import math; assert math.isnan(c.der)


def test_runconfig_whisperx_fields_default():
    cfg = asr_bench.RunConfig(device="cpu", compute_type="int8")
    assert cfg.whisperx_python is None
    assert cfg.diarize is True
    assert cfg.hf_token is None
    assert cfg.min_speakers is None and cfg.max_speakers is None
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_result.py -k "speaker_fields or runconfig_whisperx" -v` → FAIL.

- [ ] **Step 3a: ClipResult** — after `insertions: int = 0` (line ~607), before `cue_count`:

```python
    speaker_segments: List[Tuple[float, float, str]] = field(default_factory=list)
    num_speakers: int = 0
    der: float = float("nan")
```

(`field` is already imported.)

- [ ] **Step 3b: RunConfig** — after the `# nim only` block (~683), add:

```python
    # whisperx only
    whisperx_python: Optional[str] = None
    diarize: bool = True
    hf_token: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
```

- [ ] **Step 4: Run** the two tests → PASS. Full suite green (existing direct `ClipResult`/`RunConfig` constructions still work because every new field has a default).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_result.py
git commit -m "feat: ClipResult speaker fields + RunConfig whisperx fields (defaulted)"
```

---

## Task 4: speaker-prefixed VTT + word sidecar writers

**Files:**
- Modify: `asr_bench.py` — WhisperX section
- Test: `tests/test_whisperx_vtt.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import json
import asr_bench
from asr_bench import WhisperXResult


def _result():
    return WhisperXResult.from_dict({
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "hello there", "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "text": "hi back", "speaker": "SPEAKER_01"},
        ],
        "words": [{"word": "hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"}],
        "speakers": ["SPEAKER_00", "SPEAKER_01"], "language": "en",
    })


def test_write_whisperx_vtt_speaker_prefixed(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    out = asr_bench.write_whisperx_vtt(audio, "LargeV3TurboWhisperx", _result())
    body = out.read_text(encoding="utf-8")
    assert "WEBVTT" in body
    assert "SPEAKER_00: hello there" in body
    assert "SPEAKER_01: hi back" in body


def test_write_whisperx_vtt_no_speaker_prefix_when_absent(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    r = WhisperXResult.from_dict({"segments": [{"start": 0, "end": 1, "text": "plain"}], "language": "en"})
    out = asr_bench.write_whisperx_vtt(audio, "M", r)
    body = out.read_text(encoding="utf-8")
    assert "plain" in body and "SPEAKER" not in body


def test_write_words_sidecar(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    out = asr_bench.write_words_sidecar(audio, "M", _result())
    assert out.name == "Lec_Words_M.json"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["word"] == "hello"
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_vtt.py -v` → FAIL.

- [ ] **Step 3: Implement** — in the WhisperX section (reuses `_fmt_vtt_time`, `_fused_base`):

```python
def write_whisperx_vtt(audio_path: Path, model_label: str, result: "WhisperXResult") -> Path:
    """WebVTT with speaker-prefixed cues (e.g. 'SPEAKER_00: text'). Named like the
    other engines' VTTs: <base>_Captions_<Model>.vtt."""
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")
    out = audio_path.parent / f"{_fused_base(audio_path)}_Captions_{safe_model}.vtt"
    lines: List[str] = ["WEBVTT", ""]
    n = 0
    for s in result.segments:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        spk = s.get("speaker")
        if spk:
            text = f"{spk}: {text}"
        n += 1
        lines.append(str(n))
        lines.append(f"{_fmt_vtt_time(float(s['start']))} --> {_fmt_vtt_time(float(s['end']))}")
        lines.append(text)
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_words_sidecar(audio_path: Path, model_label: str, result: "WhisperXResult") -> Path:
    """Write word-level timestamps (and speaker, if present) to a JSON sidecar."""
    import json
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-")
    out = audio_path.parent / f"{_fused_base(audio_path)}_Words_{safe_model}.json"
    out.write_text(json.dumps(result.words, ensure_ascii=False, indent=0), encoding="utf-8")
    return out
```

NOTE: `_fused_base` is defined later in the Fusion section. Since both are module-level functions called at runtime (not import time), ordering doesn't matter. If you prefer, this writer may be placed after `_fused_base`; either works.

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_vtt.py -v` → PASS (3). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_vtt.py
git commit -m "feat: speaker-prefixed WhisperX VTT + word-timestamp sidecar"
```

---

## Task 5: RTTM discovery helper

**Files:**
- Modify: `asr_bench.py` — WhisperX section
- Test: `tests/test_whisperx_vtt.py` (append) — reuse this file for asr_bench-side helpers

- [ ] **Step 1: Write the failing test**

```python
def test_find_rttm_present(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    rttm = tmp_path / "Lec.rttm"; rttm.write_text("x", encoding="utf-8")
    assert asr_bench.find_rttm(audio) == rttm


def test_find_rttm_absent(tmp_path):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    assert asr_bench.find_rttm(audio) is None
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_vtt.py -k find_rttm -v` → FAIL.

- [ ] **Step 3: Implement** — in the WhisperX section:

```python
def find_rttm(audio_path: Path) -> Optional[Path]:
    """Return the <base>.rttm ground-truth sidecar next to the audio, or None.
    Matches the same base as the VTT writers (strips a trailing _default)."""
    cand = audio_path.parent / f"{_fused_base(audio_path)}.rttm"
    return cand if cand.is_file() else None
```

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_vtt.py -k find_rttm -v` → PASS (2). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_vtt.py
git commit -m "feat: find_rttm ground-truth sidecar discovery"
```

---

## Task 6: `WhisperXAdapter` ABC + Fake/InProcess/Subprocess + `make_whisperx_adapter`

**Files:**
- Modify: `asr_bench.py` — WhisperX section (needs `import importlib.util`, `subprocess`, `shutil`, `json` — `subprocess`/`shutil` already imported; add `importlib.util` if missing)
- Test: `tests/test_whisperx_adapter.py` (create)

- [ ] **Step 1: Write the failing test**

```python
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
    def fake_run(cmd, capture_output=None, text=None, timeout=None, check=None):
        calls["cmd"] = cmd
        return FakeCompleted()

    monkeypatch.setattr(asr_bench.shutil, "which", lambda n: None)
    monkeypatch.setattr(asr_bench.subprocess, "run", fake_run)
    a = asr_bench.SubprocessWhisperX(str(py))
    cfg = RunConfig(device="cpu", compute_type="int8", diarize=True, hf_token="tok")
    out = a.transcribe(str(tmp_path / "a.wav"), model="small", cfg=cfg, rttm=str(tmp_path / "a.rttm"))
    assert out.der == 0.2 and out.speakers == ["SPEAKER_00"]
    # runner script + key args present
    assert any("whisperx_runner.py" in str(c) for c in calls["cmd"])
    assert "--diarize" in calls["cmd"] and "--rttm" in calls["cmd"]
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_adapter.py -v` → FAIL.

- [ ] **Step 3: Implement** — ensure `import importlib.util` at top of `asr_bench.py` (module form, so `asr_bench.importlib.util` is patchable). Then in the WhisperX section:

```python
_RUNNER_PATH = str(Path(__file__).resolve().parent / "whisperx_runner.py")


class WhisperXAdapter(ABC):
    """Turns an audio file into a WhisperXResult. Two real impls (in-process /
    subprocess to a 3.12 venv) + a Fake for tests."""
    name: str = ""

    @abstractmethod
    def transcribe(self, audio_path: str, model: str, cfg: "RunConfig",
                   rttm: Optional[str]) -> "WhisperXResult":
        ...


class FakeWhisperXAdapter(WhisperXAdapter):
    name = "fake"

    def __init__(self, result: "WhisperXResult"):
        self._result = result

    def transcribe(self, audio_path, model, cfg, rttm):
        return self._result


def _runner_args(audio_path: str, model: str, cfg: "RunConfig", rttm: Optional[str]) -> List[str]:
    args = [_RUNNER_PATH, "--audio", audio_path, "--model", model,
            "--device", cfg.device, "--language", "en"]
    if cfg.diarize:
        args.append("--diarize")
        if cfg.hf_token:
            args += ["--hf-token", cfg.hf_token]
        if cfg.min_speakers is not None:
            args += ["--min-speakers", str(cfg.min_speakers)]
        if cfg.max_speakers is not None:
            args += ["--max-speakers", str(cfg.max_speakers)]
    if rttm:
        args += ["--rttm", rttm]
    return args


class InProcessWhisperX(WhisperXAdapter):
    """Calls whisperx_runner.run_whisperx directly (torch importable here)."""
    name = "in-process"

    def transcribe(self, audio_path, model, cfg, rttm):
        import whisperx_runner
        d = whisperx_runner.run_whisperx(
            audio=audio_path, model=model, device=cfg.device, language="en",
            diarize=cfg.diarize, hf_token=cfg.hf_token,
            min_speakers=cfg.min_speakers, max_speakers=cfg.max_speakers, rttm=rttm,
        )
        return WhisperXResult.from_dict(d)


class SubprocessWhisperX(WhisperXAdapter):
    """Runs whisperx_runner.py under a configured 3.12 venv python; parses JSON."""
    name = "subprocess"

    def __init__(self, python: str, timeout: float = 3600.0):
        self.python = python
        self.timeout = timeout

    def transcribe(self, audio_path, model, cfg, rttm):
        import json
        exe = shutil.which(self.python) or self.python
        cmd = [exe, *_runner_args(audio_path, model, cfg, rttm)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"whisperx_runner failed ({proc.returncode}): {proc.stderr[:800]}")
        return WhisperXResult.from_dict(json.loads(proc.stdout))


def _default_whisperx_python() -> Optional[str]:
    """Look for a conventional sibling venv (./.venv-whisperx)."""
    root = Path(__file__).resolve().parent
    for rel in (".venv-whisperx/Scripts/python.exe", ".venv-whisperx/bin/python"):
        cand = root / rel
        if cand.is_file():
            return str(cand)
    return None


def make_whisperx_adapter(cfg: "RunConfig") -> WhisperXAdapter:
    """Auto-select: in-process if torch importable; else subprocess to a venv
    python (cfg.whisperx_python or a default ./.venv-whisperx); else error."""
    if importlib.util.find_spec("torch") is not None:
        return InProcessWhisperX()
    venv_py = cfg.whisperx_python or _default_whisperx_python()
    if venv_py:
        return SubprocessWhisperX(venv_py)
    raise RuntimeError(
        "WhisperX needs torch (not importable here) or a 3.12 venv. Create one "
        "(py -3.12 -m venv .venv-whisperx && .venv-whisperx/Scripts/pip install whisperx) "
        "and pass --whisperx-python, or run asr-bench under a torch-enabled interpreter."
    )
```

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_adapter.py -v` → PASS (5). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_adapter.py
git commit -m "feat: WhisperXAdapter (fake/in-process/subprocess) + auto-select factory"
```

---

## Task 7: `whisperx_runner.py` (standalone core + DER + CLI)

**Files:**
- Create: `whisperx_runner.py`
- Test: `tests/test_whisperx_runner.py` (create) — tests the torch-free parts (RTTM parse, DER via importorskip, arg parsing, JSON shaping)

- [ ] **Step 1: Write the failing test**

```python
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
    # hypothesis identical to reference -> DER == 0.0
    hyp = [(0.0, 2.0, "A"), (2.0, 4.0, "B")]
    assert abs(wr.compute_der_from_rttm(hyp, str(p))) < 1e-9


def test_compute_der_all_wrong(tmp_path):
    pytest.importorskip("pyannote.metrics")
    p = tmp_path / "file.rttm"; p.write_text(RTTM, encoding="utf-8")
    # hypothesis says one speaker for the whole 4s where ref has two 2s turns
    # -> 2s confusion out of 4s ref = DER 0.5
    hyp = [(0.0, 4.0, "X")]
    der = wr.compute_der_from_rttm(hyp, str(p))
    assert abs(der - 0.5) < 1e-6
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_runner.py -v` → FAIL (module missing). (DER tests will SKIP if pyannote.metrics absent — that's expected on the 3.14 box.)

- [ ] **Step 3: Implement** `whisperx_runner.py`:

```python
#!/usr/bin/env python
"""Standalone WhisperX runner. Run under a torch-enabled (3.12) venv:

    python whisperx_runner.py --audio a.wav --model large-v3-turbo --device cuda \
        --diarize --hf-token <tok> [--rttm a.rttm]

Prints one JSON document on stdout:
  {"segments":[{start,end,text,speaker?}], "words":[...], "speakers":[...],
   "der": <float|null>, "language": "en"}

All heavy imports (torch/whisperx/pyannote) are INSIDE functions so the
torch-free helpers (parse_rttm, compute_der_from_rttm, arg parsing) import
cleanly anywhere for testing.
"""
import argparse
import json
import sys
from typing import List, Optional, Tuple


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="WhisperX transcribe+align+diarize → JSON")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="en")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--rttm", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    return ap


def parse_rttm(path: str) -> List[Tuple[float, float, str]]:
    """Parse NIST RTTM 'SPEAKER <uri> <chan> <start> <dur> <NA> <NA> <spk> ...'."""
    out: List[Tuple[float, float, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3]); dur = float(parts[4]); spk = parts[7]
            out.append((start, start + dur, spk))
    return out


def _segments_to_annotation(segments: List[Tuple[float, float, str]]):
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for start, end, spk in segments:
        if end > start:
            ann[Segment(start, end)] = spk
    return ann


def compute_der_from_rttm(hyp_segments: List[Tuple[float, float, str]], rttm_path: str) -> float:
    """DER of hyp_segments vs the RTTM reference, via pyannote.metrics (default
    collar/skip). pyannote.metrics has no torch dependency."""
    from pyannote.metrics.diarization import DiarizationErrorRate
    ref = _segments_to_annotation(parse_rttm(rttm_path))
    hyp = _segments_to_annotation(hyp_segments)
    return float(DiarizationErrorRate()(ref, hyp))


def run_whisperx(audio: str, model: str, device: str, language: str = "en",
                 diarize: bool = True, hf_token: Optional[str] = None,
                 min_speakers: Optional[int] = None, max_speakers: Optional[int] = None,
                 rttm: Optional[str] = None, batch_size: int = 16) -> dict:
    """Transcribe → align → (optional) diarize → (optional) DER. Returns a dict
    ready to JSON-serialize. Heavy imports are local."""
    import whisperx

    compute_type = "float16" if device == "cuda" else "int8"
    asr = whisperx.load_model(model, device, compute_type=compute_type, language=language)
    audio_arr = whisperx.load_audio(audio)
    result = asr.transcribe(audio_arr, batch_size=batch_size)
    lang = result.get("language", language)

    align_model, metadata = whisperx.load_align_model(language_code=lang, device=device)
    result = whisperx.align(result["segments"], align_model, metadata, audio_arr, device,
                            return_char_alignments=False)

    speakers: List[str] = []
    diarized = False
    if diarize and hf_token:
        try:
            try:
                from whisperx.diarize import DiarizationPipeline  # newer layout
            except Exception:
                from whisperx import DiarizationPipeline           # older layout
            dia = DiarizationPipeline(use_auth_token=hf_token, device=device)
            kw = {}
            if min_speakers is not None:
                kw["min_speakers"] = min_speakers
            if max_speakers is not None:
                kw["max_speakers"] = max_speakers
            diar_segments = dia(audio_arr, **kw)
            result = whisperx.assign_word_speakers(diar_segments, result)
            diarized = True
        except Exception as e:
            print(f"WARN: diarization failed ({e}); returning alignment-only", file=sys.stderr)

    segments = [{"start": float(s["start"]), "end": float(s["end"]),
                 "text": s.get("text", ""), "speaker": s.get("speaker")}
                for s in result.get("segments", [])]
    words = [{"word": w.get("word"), "start": w.get("start"), "end": w.get("end"),
              "score": w.get("score"), "speaker": w.get("speaker")}
             for w in result.get("word_segments", result.get("words", []))]
    if diarized:
        speakers = sorted({s["speaker"] for s in segments if s.get("speaker")})

    der = None
    if rttm:
        hyp = [(s["start"], s["end"], s["speaker"]) for s in segments if s.get("speaker")]
        if hyp:
            try:
                der = compute_der_from_rttm(hyp, rttm)
            except Exception as e:
                print(f"WARN: DER computation failed ({e})", file=sys.stderr)

    return {"segments": segments, "words": words, "speakers": speakers,
            "der": der, "language": lang}


def main() -> int:
    ns = build_arg_parser().parse_args()
    out = run_whisperx(
        audio=ns.audio, model=ns.model, device=ns.device, language=ns.language,
        diarize=ns.diarize, hf_token=ns.hf_token, min_speakers=ns.min_speakers,
        max_speakers=ns.max_speakers, rttm=ns.rttm, batch_size=ns.batch_size,
    )
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_runner.py -v` → `parse_rttm` + arg-parser PASS; the two DER tests PASS if `pyannote.metrics` is importable, else SKIP. Full suite green.

- [ ] **Step 5: Commit**

```bash
git add whisperx_runner.py tests/test_whisperx_runner.py
git commit -m "feat: whisperx_runner.py — transcribe/align/diarize core + DER + CLI"
```

---

## Task 8: `WhisperXEngine(Engine)` + registry

**Files:**
- Modify: `asr_bench.py` — WhisperX section + `ENGINES` (~1101)
- Test: `tests/test_whisperx_engine.py` (create)

- [ ] **Step 1: Write the failing test**

```python
import math
import asr_bench
from asr_bench import RunConfig, Pair, WhisperXResult


def test_whisperx_engine_builds_modelresult(tmp_path, monkeypatch):
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello there hi back", encoding="utf-8")

    canned = WhisperXResult.from_dict({
        "segments": [{"start": 0, "end": 2, "text": "hello there", "speaker": "SPEAKER_00"},
                     {"start": 2, "end": 4, "text": "hi back", "speaker": "SPEAKER_01"}],
        "words": [], "speakers": ["SPEAKER_00", "SPEAKER_01"], "der": 0.1, "language": "en",
    })
    # Force the engine to use a fake adapter and a known audio duration.
    monkeypatch.setattr(asr_bench, "make_whisperx_adapter", lambda cfg: asr_bench.FakeWhisperXAdapter(canned))
    monkeypatch.setattr(asr_bench, "find_rttm", lambda p: "dummy.rttm")  # presence → der kept
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 4.0)

    entry = asr_bench.resolve_model_entry("large-v3-turbo+whisperx")
    cfg = RunConfig(device="cpu", compute_type="int8", diarize=True, hf_token="tok")
    mr = asr_bench.WhisperXEngine().run(entry, [Pair(audio=audio, reference=ref)], cfg)

    assert mr.engine == "whisperx" and len(mr.clips) == 1
    c = mr.clips[0]
    assert c.num_speakers == 2
    assert c.der == 0.1
    assert c.speaker_segments[0] == (0.0, 2.0, "SPEAKER_00")
    # WER computed against the reference text (perfect match here)
    assert abs(c.wer) < 1e-9
    # a speaker-labeled VTT was written next to the audio
    assert (tmp_path / "Lec_Captions_LargeV3TurboWhisperx.vtt").exists() or \
           any(p.name.endswith(".vtt") for p in tmp_path.iterdir())


def test_whisperx_registered():
    assert "whisperx" in asr_bench.ENGINES
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_engine.py -v` → FAIL.

- [ ] **Step 3: Implement** — add an audio-duration helper + the engine in the WhisperX section. (`_audio_duration_sec` uses faster_whisper's decoder is overkill; use a lightweight probe via `whisperx`/`wave`? Simplest portable: reuse the existing decode path's duration. Implement via `av` if available, else fall back to the WhisperXResult's last segment end.)

```python
def _audio_duration_sec(audio_path: str) -> float:
    """Best-effort audio duration in seconds. Tries pyav (already a faster-whisper
    dep); falls back to 0.0 (caller then uses last-segment end)."""
    try:
        import av
        with av.open(audio_path) as container:
            if container.duration:
                return float(container.duration) / 1_000_000.0
    except Exception:
        pass
    return 0.0


class WhisperXEngine(Engine):
    name = "whisperx"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        adapter = make_whisperx_adapter(cfg)
        print(f"\n[{entry['display']}] using WhisperX adapter: {adapter.name}", flush=True)
        result_model = ModelResult(
            model_id=entry["id"], display=entry["display"], fw_name=entry.get("fw_name", ""),
            params=entry["params"], developer=entry["developer"], languages=entry["languages"],
            notes=entry["notes"], disk_bytes=None, load_sec=0.0,
            engine="whisperx", vram_is_total=False,
        )
        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] whisperx {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)
            rttm = find_rttm(pair.audio)
            t0 = time.time()
            try:
                wx = adapter.transcribe(str(pair.audio), entry["fw_name"], cfg,
                                        str(rttm) if rttm else None)
            except Exception as e:
                print(f"  ERROR whisperx on {pair.audio.name}: {e}", file=sys.stderr)
                continue
            transcribe_sec = time.time() - t0

            hypothesis = wx.text()
            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            metrics = compute_word_metrics(ref_norm, hyp_norm)

            vtt_path = write_whisperx_vtt(pair.audio, _model_label(entry["id"]), wx)
            write_words_sidecar(pair.audio, _model_label(entry["id"]), wx)

            spk_segs = wx.speaker_segments()
            audio_sec = _audio_duration_sec(str(pair.audio)) or (
                wx.segments[-1]["end"] if wx.segments else 0.0)
            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0
            der_val = wx.der if wx.der is not None else float("nan")

            print(f"    {audio_sec:.1f}s in {transcribe_sec:.1f}s "
                  f"(RTFx {rtfx:.2f}, WER {metrics.wer*100:.1f}%, "
                  f"{len(wx.speakers)} speaker(s))", flush=True)

            result_model.clips.append(ClipResult(
                audio=pair.audio.name, audio_sec=audio_sec, transcribe_sec=transcribe_sec,
                rtfx=rtfx, vram_peak_bytes=None, hypothesis=hypothesis,
                reference_normalized=ref_norm, hypothesis_normalized=hyp_norm,
                wer=metrics.wer, mer=metrics.mer, wil=metrics.wil, hits=metrics.hits,
                substitutions=metrics.substitutions, deletions=metrics.deletions,
                insertions=metrics.insertions, cue_count=len(wx.segments),
                vtt_path=str(vtt_path), reference_origin=ref_origin, reference_label=ref_label,
                speaker_segments=spk_segs, num_speakers=len(wx.speakers), der=der_val,
            ))
        if not result_model.clips:
            result_model.notes = "ALL CLIPS FAILED — check WhisperX setup/venv/token and stderr above"
        return result_model
```

Register in `ENGINES`:

```python
ENGINES: Dict[str, type] = {
    "faster-whisper": FasterWhisperEngine,
    "nim": NimEngine,
    "whisperx": WhisperXEngine,
}
```

NOTE: `WhisperXEngine` references `make_whisperx_adapter`, `find_rttm`, `write_whisperx_vtt`, `_audio_duration_sec`, `write_words_sidecar` — all in the WhisperX section. It also uses `time` (imported), `sys` (imported), `load_reference_text`, `detect_reference_origin`, `normalize_for_wer`, `compute_word_metrics`, `_model_label` (all module-level). Place `WhisperXEngine` AFTER those WhisperX-section helpers. The `ENGINES` dict is defined after all engine classes, so registering there is fine.

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_engine.py -v` → PASS (2). Full suite green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_engine.py
git commit -m "feat: WhisperXEngine.run + ENGINES registration"
```

---

## Task 9: CLI flags + `main()` wiring

**Files:**
- Modify: `asr_bench.py` — argparse in `main()` (after the nim args), cfg construction (~2065), pre-flight (~1987 area)
- Test: `tests/test_cli.py` (append), `tests/test_whisperx_engine.py` (append an end-to-end main() test)

- [ ] **Step 1: Write the failing test** — append to `tests/test_whisperx_engine.py`:

```python
def test_main_whisperx_end_to_end(tmp_path, monkeypatch):
    import asr_bench
    audio = tmp_path / "Lec_default.mp4"; audio.write_bytes(b"x")
    ref = tmp_path / "Lec.txt"; ref.write_text("hello world", encoding="utf-8")
    canned = asr_bench.WhisperXResult.from_dict(
        {"segments": [{"start": 0, "end": 2, "text": "hello world", "speaker": "SPEAKER_00"}],
         "speakers": ["SPEAKER_00"], "der": None, "language": "en"})
    monkeypatch.setattr(asr_bench, "make_whisperx_adapter", lambda cfg: asr_bench.FakeWhisperXAdapter(canned))
    monkeypatch.setattr(asr_bench, "_audio_duration_sec", lambda p: 2.0)
    out = tmp_path / "report.md"
    monkeypatch.setattr("sys.argv", [
        "asr_bench.py", "--corpus", str(tmp_path), "--models", "small+whisperx",
        "--device", "cpu", "--no-diarize", "--output", str(out)])
    rc = asr_bench.main()
    assert rc == 0
    assert out.is_file()
    assert "WhisperX" in out.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run** `python -m pytest tests/test_whisperx_engine.py -k main_whisperx -v` → FAIL (unknown args / no whisperx flags).

- [ ] **Step 3a: argparse flags** — in `main()`, after the `--nim-ssl` argument, add:

```python
    ap.add_argument("--whisperx-python", default=None,
                    help="Path to a 3.12 venv python with whisperx+torch+pyannote "
                         "(for the subprocess adapter; auto-detects ./.venv-whisperx if omitted).")
    ap.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=True,
                    help="Run pyannote speaker diarization for whisperx models (default on). "
                         "Without an HF token it warns and falls back to alignment-only.")
    ap.add_argument("--hf-token", default=None,
                    help="HuggingFace token for pyannote diarization (else HF_TOKEN/HUGGINGFACE_TOKEN env).")
    ap.add_argument("--min-speakers", type=int, default=None, help="pyannote min speakers hint.")
    ap.add_argument("--max-speakers", type=int, default=None, help="pyannote max speakers hint.")
```

- [ ] **Step 3b: cfg construction** — in the `cfg = RunConfig(...)` call (~2065), add:

```python
        whisperx_python=args.whisperx_python,
        diarize=args.diarize,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"),
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
```

(`os` is already imported.)

- [ ] **Step 3c: pre-flight token warning** — in the pre-flight area (after the fusion pre-flight block, ~2007), add a whisperx check:

```python
    # Pre-flight: warn early if diarization is requested for whisperx models without a token.
    wants_whisperx = any(resolve_model_entry(m)["engine"] == "whisperx" for m in requested)
    if wants_whisperx and args.diarize and not (
            args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")):
        print("WARNING: --diarize is on but no HF token found (--hf-token / HF_TOKEN). "
              "WhisperX will run alignment-only. Get a free token and accept the gated "
              "pyannote/speaker-diarization-3.1 model to enable diarization.", file=sys.stderr)
```

- [ ] **Step 4: Run** `python -m pytest tests/test_whisperx_engine.py -v` and `python -m pytest -q` → green. Also `python asr_bench.py --help` exits 0 with the new flags.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_whisperx_engine.py
git commit -m "feat: wire --whisperx-python/--diarize/--hf-token/--min/max-speakers into main()"
```

---

## Task 10: report DER% + Speakers columns

**Files:**
- Modify: `asr_bench.py` — `render_markdown` (headline + per-clip tables, ~1601+)
- Test: `tests/test_render.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_render.py`:

```python
def _whisperx_result():
    clip = asr_bench.ClipResult(
        audio="lec.mp4", audio_sec=600.0, transcribe_sec=20.0, rtfx=30.0,
        vram_peak_bytes=None, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.10, mer=0.09, wil=0.12,
        hits=90, substitutions=5, deletions=3, insertions=2,
        num_speakers=2, der=0.15,
        speaker_segments=[(0.0, 300.0, "SPEAKER_00"), (300.0, 600.0, "SPEAKER_01")],
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="large-v3-turbo+whisperx", display="Whisper Large V3 Turbo + WhisperX",
        fw_name="large-v3-turbo", params="809M", developer="OpenAI", languages="99",
        notes="x", disk_bytes=None, load_sec=0.0, engine="whisperx",
        vram_is_total=False, clips=[clip])


def test_der_and_speakers_shown_when_whisperx():
    md = asr_bench.render_markdown([_whisperx_result()], Path("."), _args(), "proxy")
    assert "DER%" in md and "Speakers" in md
    assert "15.0" in md   # der 0.15 -> 15.0


def test_der_absent_for_plain_whisper():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "DER%" not in md
```

- [ ] **Step 2: Run** `python -m pytest tests/test_render.py -k "der_and_speakers or der_absent" -v` → FAIL.

- [ ] **Step 3: Implement** — in `render_markdown`, compute a flag near the top (after `results` is known):

```python
    any_diar = any(
        (not math.isnan(c.der)) or c.num_speakers > 0
        for r in results for c in r.clips
    )
```

In the **headline** table, append a `DER%` + `Speakers` column to the header/separator and each row **only when `any_diar`**. Concretely, build the trailing cells conditionally:

```python
    diar_hdr = " DER% | Speakers |" if any_diar else ""
    diar_sep = "---|---|" if any_diar else ""
    lines.append("| Model | Params | Disk | Overall WER% | MER% | WIL% | RTFx | Total time | Peak VRAM |" + diar_hdr + " Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|" + diar_sep + "---|")
    for r in results:
        ...
        diar_cells = ""
        if any_diar:
            der_vals = [c.der for c in r.clips if not math.isnan(c.der)]
            der_avg = _fmt_pct(sum(der_vals) / len(der_vals)) if der_vals else "—"
            spk = max((c.num_speakers for c in r.clips), default=0) or "—"
            diar_cells = f" {der_avg} | {spk} |"
        lines.append(f"| {r.display} | {r.params} | {disk} | {wer_pct} | {mer_pct} | {wil_pct} | {rtfx} | {wall_clock} | {vram} |{diar_cells} {r.notes} |")
```

Add a short note after the headline (only when `any_diar`):

```python
    if any_diar:
        lines.append("")
        lines.append("> **Diarization:** speaker labels are pyannote hypotheses. **DER%** is "
                     "shown only for clips with an `<base>.rttm` ground-truth sidecar (pyannote.metrics "
                     "defaults). Speakers = detected speaker count.")
        lines.append("")
```

(Per-clip and per-model tables may stay WER/MER/WIL-only to limit churn; the headline DER%/Speakers + the note satisfy the spec's "shown only when present" requirement. Keep it minimal.)

- [ ] **Step 4: Run** `python -m pytest tests/test_render.py -v` and `python -m pytest -q` → green.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_render.py
git commit -m "feat: report DER% + Speakers columns (only when diarization present)"
```

---

## Task 11: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `SPEC.md`

- [ ] **Step 1: `README.md`** — add a "WhisperX (word alignment + diarization)" section:
  - What it adds (word timestamps, speaker labels, DER).
  - **Setup:** `py -3.12 -m venv .venv-whisperx` then `.venv-whisperx\Scripts\pip install whisperx`; the auto-detect of `./.venv-whisperx`; `--whisperx-python` override; in-process when torch is importable.
  - **Auth:** free HF token + accept gated `pyannote/speaker-diarization-3.1`; `--hf-token`/`HF_TOKEN`; `--no-diarize` for no-auth alignment-only; missing-token → alignment-only fallback.
  - **DER:** drop a `<base>.rttm` next to the audio to score diarization.
  - Examples: `--models large-v3-turbo+whisperx --diarize` and an `--whisperx-python` form.

- [ ] **Step 2: `CLAUDE.md`** — add to Status: WhisperX engine (alignment + diarization + DER), `<size>+whisperx` models, the 3.12-venv/in-process auto-detect, new flags. Add a "WhisperX setup notes (reference machine)" subsection (3.12 venv path, torch CUDA, HF token). Decision-log entries from the spec.

- [ ] **Step 3: `SPEC.md`** — move "WhisperX + diarization" from Planned (v0.3) to Shipped; note DER is gated on RTTM; reference `docs/superpowers/specs/2026-06-04-whisperx-diarization-design.md`; leave CER/hallucination/latency/JSON/pip as the remaining v0.3+ items.

- [ ] **Step 4: Verify** `python -m pytest -q` still green (docs only).

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md SPEC.md
git commit -m "docs: v0.3 WhisperX + diarization — setup, auth, DER, decision log"
```

---

## Task 12: Setup helper script (optional convenience)

**Files:**
- Create: `setup_whisperx_venv.ps1` (PowerShell, reference-machine convenience)
- Test: none (a shell script; verified by Task 13's live run)

- [ ] **Step 1: Write the script**

```powershell
# Creates the WhisperX venv asr-bench auto-detects (./.venv-whisperx).
# Usage:  ./setup_whisperx_venv.ps1
py -3.12 -m venv .venv-whisperx
.\.venv-whisperx\Scripts\python.exe -m pip install --upgrade pip
.\.venv-whisperx\Scripts\pip.exe install whisperx
Write-Host "Done. asr-bench auto-detects .venv-whisperx. For diarization, set HF_TOKEN and accept"
Write-Host "the gated pyannote/speaker-diarization-3.1 model on HuggingFace."
```

- [ ] **Step 2: Commit**

```bash
git add setup_whisperx_venv.ps1
git commit -m "chore: setup_whisperx_venv.ps1 convenience script"
```

---

## Task 13: LIVE integration & validation (interactive — needs HF token + multi-speaker clip)

**Not a subagent task.** Run with the user. Prerequisites the user provides: a HuggingFace token (`HF_TOKEN`) with the gated `pyannote/speaker-diarization-3.1` model accepted, and a short multi-speaker clip + `<base>.rttm`.

- [ ] **Step 1: Create the venv** — `./setup_whisperx_venv.ps1` (installs torch+whisperx+pyannote into `.venv-whisperx`, multi-GB). Verify: `.venv-whisperx\Scripts\python -c "import whisperx, torch; print(torch.cuda.is_available())"` → `True`.
- [ ] **Step 2: Smoke the runner directly** on a short lecture clip (alignment-only, no token):
  `.venv-whisperx\Scripts\python whisperx_runner.py --audio "<clip>" --model large-v3-turbo --device cuda` → inspect JSON (segments + word timestamps; `speakers` empty; `der` null).
- [ ] **Step 3: Diarized run via asr-bench** (subprocess adapter auto-detected; token set):
  `$env:HF_TOKEN="<tok>"; python asr_bench.py --models large-v3-turbo+whisperx --include "Week 16 - Friday" --limit 1` → expect a speaker-labeled `_Captions_<...>WhisperX.vtt`, ~1 speaker on single-speaker lecture, report shows Speakers column.
- [ ] **Step 4: DER run** on the user's multi-speaker clip: drop `<base>.rttm` next to it, run whisperx on that clip, confirm a **DER%** renders. Sanity-check the value is plausible.
- [ ] **Step 5: Record findings** — update the memory note `whisperx-validation.md` (live status, any version/API fixes needed in `whisperx_runner.py`, DER value observed). Commit any runner fixes the live run required.

---

## Final verification

- [ ] `python -m pytest -q` — all green (existing 91 + ~25 new; DER tests skip if pyannote.metrics absent on the 3.14 box).
- [ ] `python asr_bench.py --help` exits 0 with the whisperx flags.
- [ ] Tasks 1–12 are fully exercised without torch (Fake adapter / mocks / importorskip). Task 13 covers the live path.
- [ ] Use superpowers:requesting-code-review before merging `feat/whisperx-diarization`.
