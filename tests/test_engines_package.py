"""Phase 0 — lock the engines/ split: public surface byte-stable, registry intact,
core stays torch-free."""
import sys
import asr_bench


def test_engines_registry_has_all_families():
    assert set(asr_bench.ENGINES) == {"faster-whisper", "nim", "whisperx", "nemo", "hf"}


def test_relocated_names_still_resolve_from_asr_bench():
    for name in ("Engine", "ClipResult", "ModelResult", "RunConfig", "Pair",
                 "NeMoResult", "WhisperXResult", "NeMoEngine", "WhisperXEngine",
                 "FasterWhisperEngine", "NimEngine", "make_nemo_adapter",
                 "make_whisperx_adapter", "FakeNeMoAdapter", "compute_word_metrics",
                 "normalize_for_wer", "_audio_duration_sec", "_model_label"):
        assert hasattr(asr_bench, name), f"asr_bench.{name} missing after split"


def test_importing_asr_bench_does_not_import_torch():
    assert "torch" not in sys.modules


def test_engine_modules_are_torch_free_at_import():
    import importlib
    for mod in ("engines.base", "engines.faster_whisper", "engines.nim",
                "engines.whisperx", "engines.nemo"):
        importlib.import_module(mod)
    assert "torch" not in sys.modules
