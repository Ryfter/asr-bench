"""asr-bench engine families. Import side effect: assembles the ENGINES registry.

Kept torch-free at module scope -- heavy imports (torch/transformers/nemo/whisperx)
live inside each engine's run()/adapter, never here. asr_bench re-exports ENGINES
and the engine classes so the public surface stays byte-stable across the split."""

ENGINES: dict = {}

from engines.faster_whisper import FasterWhisperEngine  # noqa: E402
ENGINES["faster-whisper"] = FasterWhisperEngine

from engines.nim import NimEngine  # noqa: E402
ENGINES["nim"] = NimEngine
