# NVIDIA NIM (Canary) Engine Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NVIDIA NIM ASR (Riva gRPC) as a second engine family in asr-bench, validated against a self-hosted Canary NIM, with a configurable endpoint URL so the same client reaches a hosted endpoint later.

**Architecture:** Introduce a small `Engine` contract (`run(entry, pairs, cfg) -> ModelResult`) with two implementations — `FasterWhisperEngine` (a behavior-preserving refactor of today's `run_model`) and `NimEngine` (new). A `RunConfig` dataclass carries shared + engine-specific settings; an `ENGINES` registry dispatches by an `engine` field added to `MODELS`. Everything stays in `asr_bench.py`; the `engines/` package split is deferred until WhisperX/NeMo land. Pure helpers (model resolution, auth kwargs, audio decode, response parsing, cue grouping) are unit-tested with pytest; the gRPC integration is validated by live smoke runs against the Canary NIM.

**Tech Stack:** Python 3.14, `nvidia-riva-client` (gRPC, new dep), `pyav` (decode, already present), `pynvml` (VRAM), `jiwer` (WER), `pytest` 8.3 (already installed).

**Reference spec:** `docs/superpowers/specs/2026-05-30-nim-engine-support-design.md`

**Branch:** `feat/nim-engine-support` (already created).

**Conventions for every task below:**
- Run all `pytest` and `python` commands from the repo root `D:\Dev\asr-bench`.
- The existing 4 Whisper models must keep producing identical reports — several tasks are refactors and must not change Whisper behavior.

---

## Task 0: Test scaffolding

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Create the test package marker**

Create `tests/__init__.py` (empty file):

```python
```

- [ ] **Step 2: Create conftest so `import asr_bench` works from anywhere**

Create `tests/conftest.py`:

```python
import sys
from pathlib import Path

# Make the repo-root asr_bench.py importable regardless of pytest's CWD.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: Create pytest.ini**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra
```

- [ ] **Step 4: Write a smoke test that the module imports**

Create `tests/test_smoke.py`:

```python
def test_import_asr_bench():
    import asr_bench
    assert hasattr(asr_bench, "MODELS")

def test_models_registry_nonempty():
    import asr_bench
    assert "small" in asr_bench.MODELS
    assert "large-v3-turbo" in asr_bench.MODELS
```

- [ ] **Step 5: Run the smoke test**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/__init__.py tests/conftest.py pytest.ini tests/test_smoke.py
git commit -m "test: bootstrap pytest scaffolding"
```

---

## Task 1: Add `engine` field to the model registry

**Files:**
- Modify: `asr_bench.py` (the `MODELS` dict, ~lines 101-134)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_registry.py`:

```python
import asr_bench

def test_all_builtin_models_are_faster_whisper():
    for model_id, entry in asr_bench.MODELS.items():
        assert entry.get("engine") == "faster-whisper", model_id
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL — `entry.get("engine")` is `None` (key not present yet).

- [ ] **Step 3: Add `"engine": "faster-whisper"` to each MODELS entry**

In `asr_bench.py`, add `"engine": "faster-whisper",` as the first key inside each of the four model dicts (`small`, `medium`, `large-v3`, `large-v3-turbo`). Example for `small`:

```python
    "small": {
        "engine": "faster-whisper",
        "display": "Whisper Small",
        "params": "244M",
        "developer": "OpenAI",
        "languages": "99 (multilingual)",
        "fw_name": "small",
        "notes": "Real-time on CPU. Decent for clear single speaker.",
    },
```

Repeat the `"engine": "faster-whisper",` line for `medium`, `large-v3`, and `large-v3-turbo`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_registry.py
git commit -m "feat: add engine field to model registry"
```

---

## Task 2: Seed the `canary-nim` registry entry

**Files:**
- Modify: `asr_bench.py` (the `MODELS` dict — add a new entry after `large-v3-turbo`)
- Test: `tests/test_registry.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_registry.py`:

```python
def test_canary_nim_entry_present_and_shaped():
    entry = asr_bench.MODELS["canary-nim"]
    assert entry["engine"] == "nim"
    assert entry["developer"] == "NVIDIA"
    assert "riva_model" in entry  # default "" => server default
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/test_registry.py::test_canary_nim_entry_present_and_shaped -v`
Expected: FAIL — `KeyError: 'canary-nim'`.

- [ ] **Step 3: Add the entry**

In `asr_bench.py`, add to `MODELS` right after the `large-v3-turbo` entry:

```python
    "canary-nim": {
        "engine": "nim",
        "display": "Canary (NIM)",
        "params": "—",
        "developer": "NVIDIA",
        "languages": "en (+multi)",
        "riva_model": "",  # "" => let the NIM server pick its default model
        "notes": "NVIDIA NIM ASR via Riva gRPC. Endpoint set by --nim-url.",
    },
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_registry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_registry.py
git commit -m "feat: seed canary-nim registry entry"
```

---

## Task 3: `resolve_model_entry` (registry lookup + ad-hoc `nim:<name>`)

**Files:**
- Modify: `asr_bench.py` (add function after the `MODELS` dict, ~after line 134)
- Test: `tests/test_resolve.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_resolve.py`:

```python
import pytest
import asr_bench

def test_resolve_builtin_whisper():
    entry = asr_bench.resolve_model_entry("small")
    assert entry["id"] == "small"
    assert entry["engine"] == "faster-whisper"
    assert entry["fw_name"] == "small"

def test_resolve_canary_nim():
    entry = asr_bench.resolve_model_entry("canary-nim")
    assert entry["id"] == "canary-nim"
    assert entry["engine"] == "nim"
    assert entry["riva_model"] == ""

def test_resolve_adhoc_nim():
    entry = asr_bench.resolve_model_entry("nim:parakeet-1.1b")
    assert entry["id"] == "nim:parakeet-1.1b"
    assert entry["engine"] == "nim"
    assert entry["riva_model"] == "parakeet-1.1b"
    assert entry["display"] == "NIM (parakeet-1.1b)"

def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        asr_bench.resolve_model_entry("not-a-model")

def test_resolve_empty_adhoc_raises():
    with pytest.raises(ValueError):
        asr_bench.resolve_model_entry("nim:")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_resolve.py -v`
Expected: FAIL — `AttributeError: module 'asr_bench' has no attribute 'resolve_model_entry'`.

- [ ] **Step 3: Implement `resolve_model_entry`**

Add to `asr_bench.py` after the `MODELS` dict:

```python
_NIM_ADHOC_RE = re.compile(r"^nim:(.+)$")


def resolve_model_entry(model_id: str) -> Dict:
    """Resolve a --models token to a full engine entry.

    Returns a dict that always carries: id, engine, display, developer, params,
    languages, notes. NIM entries also carry riva_model; whisper entries carry
    fw_name. Raises ValueError for unknown ids.
    """
    if model_id in MODELS:
        entry = dict(MODELS[model_id])
        entry.setdefault("engine", "faster-whisper")
        entry["id"] = model_id
        return entry
    m = _NIM_ADHOC_RE.match(model_id)
    if m:
        name = m.group(1).strip()
        if not name:
            raise ValueError(f"empty NIM model name in '{model_id}'")
        return {
            "id": model_id,
            "engine": "nim",
            "display": f"NIM ({name})",
            "developer": "NVIDIA",
            "params": "—",
            "languages": "—",
            "riva_model": name,
            "notes": f"Ad-hoc NIM model '{name}' via Riva gRPC.",
        }
    raise ValueError(f"unknown model id: {model_id}")
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_resolve.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_resolve.py
git commit -m "feat: resolve_model_entry with ad-hoc nim:<name> ids"
```

---

## Task 4: `RunConfig` dataclass + `Engine` ABC + `ENGINES` registry skeleton

**Files:**
- Modify: `asr_bench.py` (add after the `Pair` dataclass / before `run_model`, ~line 268)
- Test: `tests/test_engine_contract.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_engine_contract.py`:

```python
import asr_bench

def test_runconfig_defaults():
    cfg = asr_bench.RunConfig(device="cpu", compute_type="int8")
    assert cfg.batch_size == 1
    assert cfg.beam_size == 5
    assert cfg.vad_filter is True
    assert cfg.nim_url == "localhost:50051"
    assert cfg.nim_language == "en-US"
    assert cfg.nim_api_key is None
    assert cfg.nim_ssl is False

def test_engines_registry_has_both():
    assert set(asr_bench.ENGINES.keys()) == {"faster-whisper", "nim"}
    # Each value is an Engine subclass
    for cls in asr_bench.ENGINES.values():
        assert issubclass(cls, asr_bench.Engine)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_engine_contract.py -v`
Expected: FAIL — `RunConfig` / `ENGINES` / `Engine` not defined.

- [ ] **Step 3: Add imports for ABC**

In `asr_bench.py`, add to the imports block near the top (after `from dataclasses import ...`):

```python
from abc import ABC, abstractmethod
```

- [ ] **Step 4: Define `RunConfig` and `Engine`**

Add to `asr_bench.py` immediately before the `run_model` function (currently ~line 474):

```python
@dataclass
class RunConfig:
    """Settings passed to an engine's run(). Engine-specific fields are ignored
    by the engine they don't apply to."""
    # shared
    device: str
    compute_type: str
    # faster-whisper only
    batch_size: int = 1
    beam_size: int = 5
    vad_filter: bool = True
    # nim only
    nim_url: str = "localhost:50051"
    nim_model: str = ""
    nim_language: str = "en-US"
    nim_api_key: Optional[str] = None
    nim_ssl: bool = False


class Engine(ABC):
    """Contract every ASR engine family implements. Returns a ModelResult so the
    report renderer is engine-agnostic."""
    name: str = ""

    @abstractmethod
    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> "ModelResult":
        ...
```

> Note: `ENGINES` is defined at the end of Task 6, after both engine classes exist. For now the `test_engines_registry_has_both` test will still fail — that is expected and resolved in Task 6. Run only the `RunConfig` test in the next step.

- [ ] **Step 5: Run the RunConfig test to verify it passes**

Run: `python -m pytest tests/test_engine_contract.py::test_runconfig_defaults -v`
Expected: PASS. (The registry test stays red until Task 6.)

- [ ] **Step 6: Commit**

```bash
git add asr_bench.py tests/test_engine_contract.py
git commit -m "feat: add RunConfig dataclass and Engine ABC"
```

---

## Task 5: Refactor `run_model` into `FasterWhisperEngine` (behavior-preserving)

**Files:**
- Modify: `asr_bench.py` (wrap the existing `run_model` body; add `engine`/`vram_is_total` fields to `ModelResult`)
- Test: `tests/test_engine_contract.py`

- [ ] **Step 1: Add `engine` and `vram_is_total` fields to `ModelResult`**

In `asr_bench.py`, in the `ModelResult` dataclass, add two fields after `disk_bytes`:

```python
@dataclass
class ModelResult:
    model_id: str
    display: str
    fw_name: str
    params: str
    developer: str
    languages: str
    notes: str
    disk_bytes: Optional[int]
    load_sec: float
    engine: str = "faster-whisper"
    vram_is_total: bool = False
    clips: List[ClipResult] = field(default_factory=list)
```

- [ ] **Step 2: Write the failing test for the engine wrapper**

Append to `tests/test_engine_contract.py`:

```python
def test_faster_whisper_engine_name():
    eng = asr_bench.FasterWhisperEngine()
    assert eng.name == "faster-whisper"
    assert isinstance(eng, asr_bench.Engine)
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_engine_contract.py::test_faster_whisper_engine_name -v`
Expected: FAIL — `FasterWhisperEngine` not defined.

- [ ] **Step 4: Wrap `run_model` in `FasterWhisperEngine`**

Replace the `run_model` function definition line and signature with a class method. Concretely:

1. Add this class definition immediately above the current `run_model` (keep the existing body, re-indented one level, as the method body):

```python
class FasterWhisperEngine(Engine):
    name = "faster-whisper"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        model_id = entry["id"]
        info = entry
        fw_name = info["fw_name"]
        device = cfg.device
        compute_type = cfg.compute_type
        batch_size = cfg.batch_size
        beam_size = cfg.beam_size
        vad_filter = cfg.vad_filter
        # --- existing run_model body continues below, re-indented ---
```

2. Move the entire existing body of `run_model` (everything after its docstring, from `batched_note = ...` through `return result`) into the method, indented to match. Delete the old `def run_model(...):` header and its now-duplicated local-variable setup that the block above replaces (`info = MODELS[model_id]`, `fw_name = info["fw_name"]`).

3. In the two `ModelResult(...)` constructions inside the body (the load-failure path and the success path), add `engine="faster-whisper", vram_is_total=False,`.

The resulting success-path constructor should read:

```python
        result = ModelResult(
            model_id=model_id, display=info["display"], fw_name=fw_name,
            params=info["params"], developer=info["developer"],
            languages=info["languages"], notes=info["notes"],
            disk_bytes=model_disk_bytes(fw_name), load_sec=load_sec,
            engine="faster-whisper", vram_is_total=False,
        )
```

And the load-failure constructor:

```python
            return ModelResult(
                model_id=model_id, display=info["display"], fw_name=fw_name,
                params=info["params"], developer=info["developer"],
                languages=info["languages"], notes=f"LOAD FAILED: {e}",
                disk_bytes=model_disk_bytes(fw_name), load_sec=0.0,
                engine="faster-whisper", vram_is_total=False,
            )
```

- [ ] **Step 5: Run the contract tests**

Run: `python -m pytest tests/test_engine_contract.py -v`
Expected: `test_runconfig_defaults` and `test_faster_whisper_engine_name` PASS. (`test_engines_registry_has_both` still red until Task 6.)

- [ ] **Step 6: Verify the module still imports cleanly and CLI help works**

Run: `python asr_bench.py --help`
Expected: argparse help prints with no traceback (confirms the refactor didn't break module load).

- [ ] **Step 7: Verify Whisper behavior is preserved with a tiny CPU run**

Run: `python asr_bench.py --models small --device cpu --limit 1`
Expected: a normal run completes and a report is written to `report/` — same shape as before the refactor. (If no corpus is present, this step is skipped; note it in the commit and rely on the live validation in Task 13.)

- [ ] **Step 8: Commit**

```bash
git add asr_bench.py tests/test_engine_contract.py
git commit -m "refactor: wrap run_model in FasterWhisperEngine (behavior-preserving)"
```

---

## Task 6: NIM pure helpers — auth kwargs, response parsing, cue grouping

**Files:**
- Modify: `asr_bench.py` (add helpers in a new `# ---- NIM helpers ----` section before the engine classes)
- Test: `tests/test_nim_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nim_helpers.py`:

```python
import types
import asr_bench


def test_build_nim_auth_kwargs_insecure():
    kw = asr_bench.build_nim_auth_kwargs("localhost:50051", None, False)
    assert kw["uri"] == "localhost:50051"
    assert kw["use_ssl"] is False
    assert "metadata_args" not in kw or not kw["metadata_args"]


def test_build_nim_auth_kwargs_with_key_enables_ssl_and_bearer():
    kw = asr_bench.build_nim_auth_kwargs("grpc.example.com:443", "ABC123", False)
    assert kw["use_ssl"] is True
    assert ["authorization", "Bearer ABC123"] in kw["metadata_args"]


def test_build_nim_auth_kwargs_explicit_ssl_no_key():
    kw = asr_bench.build_nim_auth_kwargs("host:50051", None, True)
    assert kw["use_ssl"] is True


def _fake_response():
    # Mimic riva RecognizeResponse: results[].alternatives[0].transcript / .words[]
    Word = lambda w, s, e: types.SimpleNamespace(word=w, start_time=s, end_time=e)
    alt = types.SimpleNamespace(
        transcript="hello world. how are you",
        words=[Word("hello", 0, 400), Word("world.", 400, 900),
               Word("how", 1000, 1200), Word("are", 1200, 1400),
               Word("you", 1400, 1700)],
    )
    result = types.SimpleNamespace(alternatives=[alt])
    return types.SimpleNamespace(results=[result])


def test_nim_response_to_hypothesis():
    hyp = asr_bench.nim_response_to_hypothesis(_fake_response())
    assert hyp == "hello world. how are you"


def test_nim_response_to_words_converts_ms_to_seconds():
    words = asr_bench.nim_response_to_words(_fake_response())
    assert words[0] == (0.0, 0.4, "hello")
    assert words[1][2] == "world."


def test_group_words_into_cues_breaks_on_sentence_end():
    words = asr_bench.nim_response_to_words(_fake_response())
    cues = asr_bench.group_words_into_cues(words, max_words=12, max_span=6.0)
    # "hello world." ends a sentence -> first cue closes there
    assert cues[0][2] == "hello world."
    assert cues[0][0] == 0.0
    assert cues[0][1] == 0.9
    assert cues[1][2] == "how are you"


def test_group_words_into_cues_breaks_on_max_words():
    words = [(float(i), float(i) + 0.5, f"w{i}") for i in range(15)]
    cues = asr_bench.group_words_into_cues(words, max_words=5, max_span=999.0)
    assert len(cues) == 3
    assert cues[0][2] == "w0 w1 w2 w3 w4"


def test_group_words_into_cues_empty():
    assert asr_bench.group_words_into_cues([], 12, 6.0) == []
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_nim_helpers.py -v`
Expected: FAIL — helpers not defined.

- [ ] **Step 3: Implement the helpers**

Add to `asr_bench.py` a new section before the engine classes:

```python
# ---- NIM helpers ------------------------------------------------------------
def build_nim_auth_kwargs(url: str, api_key: Optional[str], ssl: bool) -> Dict:
    """Build kwargs for riva.client.Auth. An API key implies SSL + a Bearer
    authorization metadata header (the path a hosted endpoint needs)."""
    kw: Dict = {"uri": url, "use_ssl": bool(ssl)}
    if api_key:
        kw["use_ssl"] = True
        kw["metadata_args"] = [["authorization", f"Bearer {api_key}"]]
    return kw


def nim_response_to_hypothesis(response) -> str:
    """Concatenate the top alternative transcript across all results."""
    parts: List[str] = []
    for result in getattr(response, "results", []) or []:
        alts = getattr(result, "alternatives", None) or []
        if alts:
            parts.append(alts[0].transcript)
    return " ".join(p.strip() for p in parts if p).strip()


def nim_response_to_words(response) -> List[Tuple[float, float, str]]:
    """Flatten word-level timings (Riva reports ms) into (start_s, end_s, word)."""
    out: List[Tuple[float, float, str]] = []
    for result in getattr(response, "results", []) or []:
        alts = getattr(result, "alternatives", None) or []
        if not alts:
            continue
        for w in getattr(alts[0], "words", None) or []:
            out.append((float(w.start_time) / 1000.0, float(w.end_time) / 1000.0, w.word))
    return out


def group_words_into_cues(
    words: List[Tuple[float, float, str]],
    max_words: int = 12,
    max_span: float = 6.0,
) -> List[Tuple[float, float, str]]:
    """Group word timings into VTT-style cues. Close a cue on sentence-final
    punctuation, or when it reaches max_words, or spans >= max_span seconds."""
    cues: List[Tuple[float, float, str]] = []
    buf: List[str] = []
    start: Optional[float] = None
    end: float = 0.0
    for (ws, we, text) in words:
        if start is None:
            start = ws
        buf.append(text)
        end = we
        ends_sentence = text.rstrip().endswith((".", "?", "!"))
        if ends_sentence or len(buf) >= max_words or (end - start) >= max_span:
            cues.append((start, end, " ".join(buf)))
            buf, start = [], None
    if buf and start is not None:
        cues.append((start, end, " ".join(buf)))
    return cues
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_nim_helpers.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_nim_helpers.py
git commit -m "feat: NIM pure helpers (auth kwargs, response parsing, cue grouping)"
```

---

## Task 7: Audio decode helper `decode_to_pcm16`

**Files:**
- Modify: `asr_bench.py` (add to the NIM helpers section)
- Test: `tests/test_decode.py`

- [ ] **Step 1: Write the failing test (generates a real 1s WAV)**

Create `tests/test_decode.py`:

```python
import math
import struct
import wave
import asr_bench


def _write_sine_wav(path, seconds=1.0, rate=16000, freq=440.0):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        frames = b"".join(
            struct.pack("<h", int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(n)
        )
        wf.writeframes(frames)


def test_decode_to_pcm16_roundtrip(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_sine_wav(wav, seconds=1.0, rate=16000)
    pcm, n_samples = asr_bench.decode_to_pcm16(wav)
    assert isinstance(pcm, (bytes, bytearray))
    assert len(pcm) == n_samples * 2          # s16le => 2 bytes/sample
    # ~16000 samples for 1 second at 16kHz (allow small resampler edge slack)
    assert 15500 <= n_samples <= 16500


def test_decode_to_pcm16_resamples_44k_to_16k(tmp_path):
    wav = tmp_path / "tone44.wav"
    _write_sine_wav(wav, seconds=1.0, rate=44100)
    pcm, n_samples = asr_bench.decode_to_pcm16(wav)
    assert 15500 <= n_samples <= 16500       # resampled down to 16kHz
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_decode.py -v`
Expected: FAIL — `decode_to_pcm16` not defined.

- [ ] **Step 3: Implement `decode_to_pcm16` (pyav primary, ffmpeg fallback)**

Add to the NIM helpers section in `asr_bench.py`:

```python
def decode_to_pcm16(path: Path, target_rate: int = 16000) -> Tuple[bytes, int]:
    """Decode any audio/video file to 16kHz mono s16le PCM bytes.

    Primary: pyav (already installed via faster-whisper). Fallback: an ffmpeg
    subprocess. Returns (pcm_bytes, n_samples). Raises RuntimeError if neither
    path works.
    """
    # --- Primary: pyav ---
    try:
        import av  # type: ignore
        from av.audio.resampler import AudioResampler  # type: ignore

        container = av.open(str(path))
        resampler = AudioResampler(format="s16", layout="mono", rate=target_rate)
        chunks: List[bytes] = []
        for frame in container.decode(audio=0):
            for rframe in resampler.resample(frame):
                chunks.append(bytes(rframe.planes[0]))
        # Flush the resampler.
        for rframe in resampler.resample(None):
            chunks.append(bytes(rframe.planes[0]))
        container.close()
        pcm = b"".join(chunks)
        if pcm:
            return pcm, len(pcm) // 2
    except Exception:
        pass  # fall through to ffmpeg

    # --- Fallback: ffmpeg subprocess ---
    import subprocess
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-i", str(path),
             "-ar", str(target_rate), "-ac", "1", "-f", "s16le", "-"],
            capture_output=True, check=True,
        )
        pcm = proc.stdout
        return pcm, len(pcm) // 2
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"could not decode {path} via pyav or ffmpeg: {e}")
```

> Note on pyav resampler API: on pyav 17 `resampler.resample(frame)` returns a list of frames. The code above handles that. If a future pyav returns a single frame, wrap in a list.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_decode.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_decode.py
git commit -m "feat: decode_to_pcm16 (pyav primary, ffmpeg fallback)"
```

---

## Task 8: VRAM sampler for the single NIM RPC

**Files:**
- Modify: `asr_bench.py` (add to the NIM helpers section)
- Test: `tests/test_vram_sampler.py`

- [ ] **Step 1: Write the failing test (deterministic, no real GPU)**

Create `tests/test_vram_sampler.py`:

```python
import asr_bench


def test_vram_sampler_records_peak_of_injected_reads():
    reads = iter([100, 500, 300, 900, 200])

    def fake_read():
        try:
            return next(reads)
        except StopIteration:
            return 0

    s = asr_bench.VramSampler(read_fn=fake_read, interval=0.001)
    # Drive the recording logic deterministically rather than relying on thread timing.
    for _ in range(5):
        s._record(fake_read())
    assert s.peak == 900


def test_vram_sampler_peak_starts_at_zero():
    s = asr_bench.VramSampler(read_fn=lambda: 0, interval=0.01)
    assert s.peak == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_vram_sampler.py -v`
Expected: FAIL — `VramSampler` not defined.

- [ ] **Step 3: Implement `VramSampler`**

Add to the NIM helpers section in `asr_bench.py`:

```python
import threading  # add to the top imports block if not already present


class VramSampler:
    """Background poller that records peak total GPU memory used during a call.

    Used for the NIM path, where a single blocking offline_recognize RPC offers
    no per-segment loop to sample in. Reports TOTAL used (model is pre-resident
    in the container), not a per-clip delta — callers must mark it as such.
    """
    def __init__(self, read_fn=gpu_used_bytes, interval: float = 0.1):
        self._read_fn = read_fn
        self._interval = interval
        self.peak: int = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _record(self, value: int) -> None:
        if value > self.peak:
            self.peak = value

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._record(self._read_fn())
            self._stop.wait(self._interval)

    def start(self) -> "VramSampler":
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.peak
```

> If `import threading` is added inline above the class, also remove it from there once you confirm it's in the top imports block — keep imports at the top per the file's style.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_vram_sampler.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_vram_sampler.py
git commit -m "feat: VramSampler for peak GPU usage during the NIM RPC"
```

---

## Task 9: `NimEngine.run` + `ENGINES` registry

**Files:**
- Modify: `asr_bench.py` (add `NimEngine` after `FasterWhisperEngine`; define `ENGINES`)
- Test: `tests/test_engine_contract.py` (the registry test goes green here)

- [ ] **Step 1: Implement `NimEngine`**

Add to `asr_bench.py` after `FasterWhisperEngine`:

```python
class NimEngine(Engine):
    name = "nim"

    def run(self, entry: Dict, pairs: List[Pair], cfg: RunConfig) -> ModelResult:
        riva_model = cfg.nim_model or entry.get("riva_model", "")
        print(
            f"\n[{entry['display']}] connecting to NIM at {cfg.nim_url} "
            f"(model={riva_model or 'server default'})...",
            flush=True,
        )

        def _fail(msg: str) -> ModelResult:
            print(f"  ERROR: {msg}", file=sys.stderr)
            return ModelResult(
                model_id=entry["id"], display=entry["display"], fw_name="",
                params=entry.get("params", "—"), developer=entry.get("developer", "NVIDIA"),
                languages=entry.get("languages", "—"), notes=f"LOAD FAILED: {msg}",
                disk_bytes=None, load_sec=0.0, engine="nim", vram_is_total=True,
            )

        t0 = time.time()
        try:
            import riva.client  # late import; only needed for NIM runs
            auth = riva.client.Auth(**build_nim_auth_kwargs(cfg.nim_url, cfg.nim_api_key, cfg.nim_ssl))
            asr = riva.client.ASRService(auth)
        except ImportError:
            return _fail("nvidia-riva-client not installed (pip install nvidia-riva-client)")
        except Exception as e:
            return _fail(f"could not connect to NIM at {cfg.nim_url}: {e}")
        load_sec = time.time() - t0
        print(f"  connected in {load_sec:.1f}s", flush=True)

        result = ModelResult(
            model_id=entry["id"], display=entry["display"], fw_name="",
            params=entry.get("params", "—"), developer=entry.get("developer", "NVIDIA"),
            languages=entry.get("languages", "—"), notes=entry.get("notes", ""),
            disk_bytes=None, load_sec=load_sec, engine="nim", vram_is_total=True,
        )

        from jiwer import wer as jiwer_wer

        for clip_idx, pair in enumerate(pairs, start=1):
            print(f"  [{clip_idx}/{len(pairs)}] transcribing {pair.audio.name}...", flush=True)
            ref_text = load_reference_text(pair.reference)
            ref_origin, ref_label = detect_reference_origin(pair.reference)

            try:
                pcm, n_samples = decode_to_pcm16(pair.audio)
            except Exception as e:
                print(f"    decode failed: {e}", file=sys.stderr)
                continue
            audio_sec = n_samples / 16000.0

            config = riva.client.RecognitionConfig(
                language_code=cfg.nim_language,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                max_alternatives=1,
            )
            # encoding / sample rate: LINEAR_PCM @ 16k mono
            config.sample_rate_hertz = 16000
            config.audio_channel_count = 1
            try:
                config.encoding = riva.client.AudioEncoding.LINEAR_PCM
            except AttributeError:
                from riva.client.proto.riva_audio_pb2 import AudioEncoding  # type: ignore
                config.encoding = AudioEncoding.LINEAR_PCM
            if riva_model:
                config.model = riva_model

            sampler = VramSampler().start() if _HAS_NVML else None
            t1 = time.time()
            try:
                response = asr.offline_recognize(pcm, config)
            except Exception as e:
                if sampler:
                    sampler.stop()
                print(f"    recognize failed: {e}", file=sys.stderr)
                continue
            transcribe_sec = time.time() - t1
            vram_peak = sampler.stop() if sampler else None

            hypothesis = nim_response_to_hypothesis(response)
            words = nim_response_to_words(response)
            cue_tuples = group_words_into_cues(words)
            vtt_path = write_whisper_vtt(pair.audio, _model_label(entry["id"]), cue_tuples)

            ref_norm = normalize_for_wer(ref_text)
            hyp_norm = normalize_for_wer(hypothesis)
            try:
                wer_val = jiwer_wer(ref_norm, hyp_norm)
            except Exception:
                wer_val = float("nan")

            rtfx = audio_sec / transcribe_sec if transcribe_sec > 0 else 0.0
            print(
                f"    {audio_sec:.1f}s audio in {transcribe_sec:.1f}s "
                f"(RTFx {rtfx:.2f}, WER {wer_val * 100:.1f}%)",
                flush=True,
            )

            result.clips.append(
                ClipResult(
                    audio=pair.audio.name, audio_sec=audio_sec,
                    transcribe_sec=transcribe_sec, rtfx=rtfx,
                    vram_peak_bytes=vram_peak, hypothesis=hypothesis,
                    reference_normalized=ref_norm, hypothesis_normalized=hyp_norm,
                    wer=wer_val, cue_count=len(cue_tuples), vtt_path=str(vtt_path),
                    reference_origin=ref_origin, reference_label=ref_label,
                )
            )
        return result
```

- [ ] **Step 2: Define the `ENGINES` registry**

Add immediately after `NimEngine`:

```python
ENGINES: Dict[str, type] = {
    "faster-whisper": FasterWhisperEngine,
    "nim": NimEngine,
}
```

- [ ] **Step 3: Add a NimEngine identity test**

Append to `tests/test_engine_contract.py`:

```python
def test_nim_engine_name():
    eng = asr_bench.NimEngine()
    assert eng.name == "nim"
    assert isinstance(eng, asr_bench.Engine)
```

- [ ] **Step 4: Run the full contract test file**

Run: `python -m pytest tests/test_engine_contract.py -v`
Expected: all PASS now (including `test_engines_registry_has_both`).

- [ ] **Step 5: Commit**

```bash
git add asr_bench.py tests/test_engine_contract.py
git commit -m "feat: NimEngine.run + ENGINES dispatch registry"
```

---

## Task 10: Report rendering — n/a disk, `*` VRAM marker, engines note, reproducibility flags

**Files:**
- Modify: `asr_bench.py` (`render_markdown`)
- Test: `tests/test_render.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_render.py`:

```python
import types
from pathlib import Path
import asr_bench


def _whisper_result():
    clip = asr_bench.ClipResult(
        audio="lecture.mp4", audio_sec=600.0, transcribe_sec=10.0, rtfx=60.0,
        vram_peak_bytes=200 * 1024**2, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.10, cue_count=50,
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="large-v3-turbo", display="Whisper Large V3 Turbo",
        fw_name="large-v3-turbo", params="809M", developer="OpenAI",
        languages="99", notes="x", disk_bytes=1600 * 1024**2, load_sec=2.0,
        engine="faster-whisper", vram_is_total=False, clips=[clip],
    )


def _nim_result():
    clip = asr_bench.ClipResult(
        audio="lecture.mp4", audio_sec=600.0, transcribe_sec=8.0, rtfx=75.0,
        vram_peak_bytes=9 * 1024**3, hypothesis="hi", reference_normalized="hi",
        hypothesis_normalized="hi", wer=0.09, cue_count=48,
        reference_origin="unknown", reference_label="user-provided reference",
    )
    return asr_bench.ModelResult(
        model_id="canary-nim", display="Canary (NIM)", fw_name="", params="—",
        developer="NVIDIA", languages="en", notes="x", disk_bytes=None,
        load_sec=0.5, engine="nim", vram_is_total=True, clips=[clip],
    )


def _args():
    return types.SimpleNamespace(
        device="cuda", compute_type="float16", batch_size=1, beam_size=5,
        vad_filter=True, models=["large-v3-turbo", "canary-nim"],
        nim_url="localhost:50051", nim_model="", nim_language="en-US",
        nim_api_key=None, nim_ssl=False,
    )


def test_nim_disk_renders_na():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    # The NIM headline row shows n/a for disk, not "?"
    nim_line = [l for l in md.splitlines() if l.startswith("| Canary (NIM)")][0]
    assert "n/a" in nim_line


def test_nim_vram_has_star_marker():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    nim_line = [l for l in md.splitlines() if l.startswith("| Canary (NIM)")][0]
    assert "*" in nim_line


def test_engines_note_present_when_nim_in_run():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    assert "Engines in this run" in md
    assert "total GPU memory" in md  # the footnote explaining the * marker


def test_reproducibility_includes_nim_flags():
    md = asr_bench.render_markdown([_whisper_result(), _nim_result()], Path("."), _args(), "proxy")
    assert "--nim-url localhost:50051" in md


def test_no_engines_note_for_whisper_only():
    md = asr_bench.render_markdown([_whisper_result()], Path("."), _args(), "proxy")
    assert "Engines in this run" not in md
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_render.py -v`
Expected: FAIL on the n/a, star, engines-note, and nim-flags assertions.

- [ ] **Step 3: Add a helper for the VRAM cell and use it in both tables**

In `asr_bench.py`, add near the other small helpers (e.g. after `_model_label`):

```python
def _vram_cell(value: Optional[int], is_total: bool) -> str:
    """Render a VRAM cell; mark NIM 'total used' values with a trailing '*'."""
    if value is None:
        return "n/a" if not _HAS_NVML else "0"
    return fmt_bytes(value) + ("*" if is_total else "")


def _disk_cell(result: "ModelResult") -> str:
    return "n/a" if result.engine == "nim" else fmt_bytes(result.disk_bytes)
```

- [ ] **Step 4: Use the helpers in the headline table**

In `render_markdown`, replace the headline row construction loop body:

```python
    for r in results:
        wall_clock = f"{r.total_transcribe_sec:.1f}s"
        wer_pct = f"{r.avg_wer * 100:.1f}" if r.clips else "—"
        rtfx = f"{r.aggregate_rtfx:.2f}x" if r.clips else "—"
        vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
        disk = _disk_cell(r)
        lines.append(
            f"| {r.display} | {r.params} | {disk} | {wer_pct} | {rtfx} | {wall_clock} | {vram} | {r.notes} |"
        )
```

- [ ] **Step 5: Use the VRAM helper in the per-clip and per-model tables**

In the **per-clip view** loop, replace the `vram = ...` line with:

```python
                    vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
```

In the **per-model breakdown** per-clip rows, replace the `vram = ...` line with:

```python
            vram = _vram_cell(c.vram_peak_bytes, r.vram_is_total)
```

And the **OVERALL** row's `overall_vram = ...` line with:

```python
        overall_vram = _vram_cell(r.peak_vram_bytes, r.vram_is_total)
```

- [ ] **Step 6: Add the "Engines in this run" note (only when a NIM row is present)**

In `render_markdown`, immediately after the headline table block (after its trailing `lines.append("")`), add:

```python
    # ---- Engines note: explain metric-fidelity differences when NIM is present ----
    if any(r.engine == "nim" for r in results):
        lines.append("## Engines in this run")
        lines.append("")
        lines.append(
            "This run mixes engine families. The in-process **faster-whisper** engine "
            "returns the fuller, more directly-comparable set of readings: a per-clip "
            "**VRAM allocation delta**, the **on-disk** model size, and weights **load** time."
        )
        lines.append("")
        lines.append(
            "The **NIM** engine runs behind a gRPC service, so it exposes less. "
            "Its VRAM figures are marked `*` = **total** GPU memory in use during the clip "
            "(the model is pre-resident in the NIM container), **not** the per-clip allocation "
            "delta that Whisper rows report — the two are not directly comparable. Disk size "
            "is shown as `n/a` (it is a container image, not an HF cache dir). NIM is still "
            "fully benchmarkable for **WER**, **RTFx**, and **wall clock** — run it and see how it does."
        )
        lines.append("")
```

- [ ] **Step 7: Echo `--nim-*` flags in the reproducibility command when NIM ran**

In `render_markdown`, find the reproducibility command line construction. Replace it with a version that appends NIM flags when any NIM model is present:

```python
    batch_flag = f" --batch-size {args.batch_size}" if args.batch_size > 1 else ""
    beam_flag = f" --beam-size {args.beam_size}" if args.beam_size != 5 else ""
    vad_flag = "" if args.vad_filter else " --no-vad-filter"
    nim_flag = ""
    if any(r.engine == "nim" for r in results):
        nim_flag = f" --nim-url {args.nim_url} --nim-language {args.nim_language}"
        if args.nim_model:
            nim_flag += f" --nim-model {args.nim_model}"
    lines.append(f"- Command: `python asr_bench.py --corpus '{corpus_path}' --models {','.join(args.models)} --device {args.device} --compute-type {args.compute_type}{batch_flag}{beam_flag}{vad_flag}{nim_flag}`")
```

- [ ] **Step 8: Run to verify pass**

Run: `python -m pytest tests/test_render.py -v`
Expected: all 5 PASS.

- [ ] **Step 9: Run the full suite for regressions**

Run: `python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add asr_bench.py tests/test_render.py
git commit -m "feat: NIM-aware report rendering (n/a disk, * VRAM marker, engines note, repro flags)"
```

---

## Task 11: CLI flags + `main()` dispatch through the engine registry

**Files:**
- Modify: `asr_bench.py` (`main()` — argparse flags, model validation, RunConfig build, dispatch loop)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test (argument parsing + validation)**

Create `tests/test_cli.py`:

```python
import subprocess
import sys


def test_help_lists_nim_flags():
    out = subprocess.run(
        [sys.executable, "asr_bench.py", "--help"],
        capture_output=True, text=True,
    ).stdout
    assert "--nim-url" in out
    assert "--nim-model" in out
    assert "--nim-language" in out
    assert "--nim-api-key" in out


def test_adhoc_nim_id_accepted_not_rejected_as_unknown():
    # A bogus corpus dir makes the run exit early, but model validation happens
    # first; an unknown-model error must NOT mention nim:foo.
    res = subprocess.run(
        [sys.executable, "asr_bench.py", "--models", "nim:foo",
         "--corpus", "does-not-exist-xyz"],
        capture_output=True, text=True,
    )
    assert "unknown models" not in (res.stdout + res.stderr).lower()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — flags absent; `nim:foo` rejected as unknown model.

- [ ] **Step 3: Add the `--nim-*` argparse flags**

In `main()`, after the `--vad-filter` argument block, add:

```python
    ap.add_argument(
        "--nim-url", default="localhost:50051",
        help="NIM/Riva gRPC endpoint for nim-engine models (self-hosted or hosted).",
    )
    ap.add_argument(
        "--nim-model", default="",
        help="Override the Riva model name for the canary-nim entry. '' = server default.",
    )
    ap.add_argument(
        "--nim-language", default="en-US",
        help="Riva language code for NIM models (note: Whisper uses 'en').",
    )
    ap.add_argument(
        "--nim-api-key", default=None,
        help="Bearer token for a secured NIM endpoint. Presence auto-enables SSL.",
    )
    ap.add_argument(
        "--nim-ssl", action="store_true",
        help="Force SSL for the NIM endpoint without an API key (e.g. self-signed TLS).",
    )
```

- [ ] **Step 4: Update model validation to accept `nim:` ids**

In `main()`, replace the unknown-model check:

```python
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = []
    for m in requested:
        try:
            resolve_model_entry(m)
        except ValueError:
            unknown.append(m)
    if unknown:
        print(f"ERROR: unknown models: {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(MODELS.keys())} (or ad-hoc 'nim:<riva-model-name>')", file=sys.stderr)
        return 2
```

- [ ] **Step 5: Build a `RunConfig` and dispatch through `ENGINES`**

In `main()`, replace the results-building loop (currently calls `run_model(...)`):

```python
    cfg = RunConfig(
        device=device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        nim_url=args.nim_url,
        nim_model=args.nim_model,
        nim_language=args.nim_language,
        nim_api_key=args.nim_api_key,
        nim_ssl=args.nim_ssl,
    )

    results: List[ModelResult] = []
    for model_id in requested:
        entry = resolve_model_entry(model_id)
        engine_cls = ENGINES.get(entry["engine"])
        if engine_cls is None:
            print(f"ERROR: no engine registered for '{entry['engine']}' (model {model_id})", file=sys.stderr)
            return 2
        results.append(engine_cls().run(entry, pairs, cfg))
```

- [ ] **Step 6: Ensure batch-size auto-resolution only considers Whisper models**

The `recommend_batch_size` path uses `_MODEL_VRAM_COST`, which only has Whisper keys — NIM ids are simply absent, so the existing `if mid in _MODEL_VRAM_COST` filter already ignores them. No change needed, but verify the `requested` list passed to `recommend_batch_size(requested)` still works when it contains `canary-nim`/`nim:...` (it will: those ids are filtered out). Leave as-is.

- [ ] **Step 7: Run the CLI tests**

Run: `python -m pytest tests/test_cli.py -v`
Expected: both PASS.

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest -v`
Expected: all PASS.

- [ ] **Step 9: Verify a NIM-only invocation fails gracefully (no endpoint needed)**

Run: `python asr_bench.py --models canary-nim --corpus test-corpus --nim-url localhost:1 --limit 1`
Expected: either a "no pairs discovered" message (if corpus empty) or, with a corpus present, a `LOAD FAILED` / connection-refused row in the report — **not** a traceback. (This exercises the graceful-failure path without a live NIM.)

- [ ] **Step 10: Commit**

```bash
git add asr_bench.py tests/test_cli.py
git commit -m "feat: --nim-* CLI flags and engine-registry dispatch in main()"
```

---

## Task 12: Documentation updates

**Files:**
- Modify: `CLAUDE.md` (deps, dev workflow, decision log)
- Modify: `SPEC.md` (decision log — NIM pulled forward, local-only nuance)
- Modify: `README.md` (install note for `nvidia-riva-client`; NIM usage example)

- [ ] **Step 1: Update CLAUDE.md — setup notes (add the new dep)**

In `CLAUDE.md`, under "Locally-discovered setup notes", add a bullet:

```markdown
- **nvidia-riva-client** (for NIM engine): `pip install nvidia-riva-client`. Lazy-imported — only required when a `nim`-engine model (e.g. `canary-nim`, `nim:<name>`) is requested. Whisper-only runs don't need it.
```

- [ ] **Step 2: Update CLAUDE.md — development workflow (Engine contract)**

In `CLAUDE.md`, under "Development workflow", replace the "Add a new engine family" bullet with:

```markdown
- **Add a new engine family**: implement the `Engine` ABC (`run(entry, pairs, cfg) -> ModelResult`), register the class in `ENGINES`, and give its models `"engine": "<name>"` in `MODELS`. `FasterWhisperEngine` and `NimEngine` are the two reference implementations. Share the metrics infrastructure (`ClipResult`, `ModelResult`, `render_markdown`). The `engines/` package split is deferred until a third family (WhisperX/NeMo) lands.
- **Add a new Whisper variant**: extend the `MODELS` dict (`"engine": "faster-whisper"`) + add an entry to `_MODEL_VRAM_COST` for batch sizing.
```

- [ ] **Step 3: Update CLAUDE.md — common workflows (NIM example)**

In `CLAUDE.md`, under "Common workflows", add a subsection:

```markdown
### Benchmark a self-hosted NIM against Whisper
```powershell
python asr_bench.py --models large-v3-turbo,canary-nim --nim-url localhost:50051
```
NIM rows report WER/RTFx/wall-clock normally; VRAM is shown as total-used (`*`) and disk as `n/a` (see the report's "Engines in this run" note). Ad-hoc unregistered NIM models: `--models nim:<riva-model-name>`.
```

- [ ] **Step 4: Update CLAUDE.md and SPEC.md decision logs**

Append to the **Decision log** in BOTH `CLAUDE.md` and `SPEC.md`:

```markdown
- **2026-05-30** — Added NVIDIA NIM ASR (Riva gRPC) as the second engine family, ahead of its v0.3 roadmap slot, validated against a self-hosted Canary NIM. Stays within the "local engines only" rule: a self-hosted NIM is local inference behind a gRPC port, not a cloud ASR API. The `--nim-url` flag *permits* a hosted endpoint, but defaults and validation are local. Introduced the `Engine` contract (`FasterWhisperEngine` + `NimEngine`) in-file; deferred the `engines/` package split until WhisperX/NeMo land.
```

- [ ] **Step 5: Update README.md**

In `README.md`, add an install note and a usage example. Add under the install/requirements section:

```markdown
### Optional: NVIDIA NIM engine

To benchmark a NIM ASR endpoint (e.g. a self-hosted Canary NIM):

```bash
pip install nvidia-riva-client
```

This is only needed when you request a `nim`-engine model. Example:

```bash
python asr_bench.py --models large-v3-turbo,canary-nim --nim-url localhost:50051
```

NIM rows report WER, RTFx, and wall-clock like any engine. Because the model
runs behind a gRPC service, VRAM is reported as *total* GPU memory in use
(marked `*`) rather than a per-clip delta, and disk size shows `n/a`. Use
`--models nim:<riva-model-name>` to benchmark an unregistered NIM model.
```

> If `README.md` has no clear install section, add these under a new `## NVIDIA NIM engine (optional)` heading near the bottom.

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md SPEC.md README.md
git commit -m "docs: document NIM engine (deps, workflow, decision log, README)"
```

---

## Task 13: Live validation against the Canary NIM

> These steps require the live self-hosted Canary NIM and `nvidia-riva-client` installed. They are manual verification, not automated tests. Record actual outputs.

- [ ] **Step 1: Install the runtime dep**

Run: `pip install nvidia-riva-client`
Expected: installs cleanly; `python -c "import riva.client; print('ok')"` prints `ok`.

- [ ] **Step 2: Confirm the NIM endpoint URL and model name**

Confirm the Canary NIM's gRPC address (default assumed `localhost:50051`) and, if needed, the Riva model name. If the address differs, use `--nim-url <host:port>`; if a specific model must be named, use `--nim-model <name>`.

- [ ] **Step 3: Smoke run — one clip, NIM only**

Run: `python asr_bench.py --models canary-nim --corpus test-corpus --limit 1`
Expected: connects, decodes the clip, transcribes via NIM, prints an RTFx/WER line, writes a `*_Captions_CanaryNim.vtt`, and saves a report. No traceback.

- [ ] **Step 4: Head-to-head — Whisper vs NIM, in one report**

Run: `python asr_bench.py --models large-v3-turbo,canary-nim --corpus test-corpus --limit 2`
Expected: the report's Headline ranks both engines; the NIM row shows `n/a` disk and a `*`-marked VRAM value; an "Engines in this run" note is present; cue-density anomaly section (if any) treats NIM cues correctly.

- [ ] **Step 5: Ad-hoc id**

Run: `python asr_bench.py --models nim:<a-model-served-by-your-nim> --corpus test-corpus --limit 1`
Expected: resolves the ad-hoc id and runs (or returns a clean `LOAD FAILED` row if that model name isn't served) — no "unknown models" error.

- [ ] **Step 6: Graceful failure**

Run: `python asr_bench.py --models canary-nim --corpus test-corpus --nim-url localhost:1 --limit 1`
Expected: a `LOAD FAILED` row in the report, not a crash.

- [ ] **Step 7: Final full test suite**

Run: `python -m pytest -v`
Expected: all automated tests PASS.

- [ ] **Step 8: Commit any fixes discovered during validation**

```bash
git add -A
git commit -m "fix: adjustments from live Canary NIM validation"
```

(If no fixes were needed, skip this commit.)

---

## Task 14: Finish the branch

- [ ] **Step 1: Confirm the full suite and a clean tree**

Run: `python -m pytest -v` and `git status`
Expected: all tests pass; working tree clean.

- [ ] **Step 2: Hand off to the finishing-a-development-branch skill**

Use the `superpowers:finishing-a-development-branch` skill to decide merge / PR / cleanup for `feat/nim-engine-support`.

---

## Self-review notes (author checklist — completed)

- **Spec coverage:** §2.1 engine field → Task 1; §2.1 canary-nim → Task 2; §2.2 ad-hoc ids → Task 3; §2.3 Engine/RunConfig/ENGINES → Tasks 4, 9; FasterWhisperEngine refactor → Task 5; §3.1 decode → Task 7; §3.2 transcribe/parse/cues → Tasks 6, 9; §3.3 no-streaming note → inherent in Task 9; §4 metrics + §4.1 VRAM sampler → Tasks 8, 9, 10; §5 CLI flags → Task 11; §6 render changes → Task 10; §7 deps → Tasks 12, 13; §8 validation → Task 13; §9 single-file → respected (no new modules). All spec sections mapped.
- **Placeholder scan:** no TBD/TODO; every code step shows complete code; test bodies are concrete.
- **Type/name consistency:** `RunConfig`, `Engine`, `ENGINES`, `resolve_model_entry`, `build_nim_auth_kwargs`, `nim_response_to_hypothesis`, `nim_response_to_words`, `group_words_into_cues`, `decode_to_pcm16`, `VramSampler`, `_vram_cell`, `_disk_cell`, `ModelResult.engine`, `ModelResult.vram_is_total` are used identically across tasks.
