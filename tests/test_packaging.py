"""B1 — packaging metadata wiring. Validates pyproject.toml without a network
install: the console entry point, py-modules, and dynamic-version source must
line up with the actual modules."""
import tomllib
from pathlib import Path

import asr_bench

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_version_is_nonempty_string():
    assert isinstance(asr_bench.__version__, str)
    assert asr_bench.__version__


def test_console_script_points_at_main():
    scripts = _pyproject()["project"]["scripts"]
    assert scripts.get("asr-bench") == "asr_bench:main"


def test_py_modules_cover_all_top_level_modules():
    mods = set(_pyproject()["tool"]["setuptools"]["py-modules"])
    assert {"asr_bench", "asr_compare", "whisperx_runner", "nemo_runner"} <= mods


def test_dynamic_version_sourced_from_module():
    data = _pyproject()
    assert "version" in data["project"].get("dynamic", [])
    attr = data["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    assert attr == "asr_bench.__version__"


def test_engines_package_declared():
    data = _pyproject()
    pkgs = data["tool"]["setuptools"].get("packages", [])
    assert "engines" in pkgs


def test_importing_asr_bench_stays_torch_free():
    # asr_bench now imports the engines package; that must not pull torch.
    import sys
    import asr_bench  # noqa: F401
    assert "torch" not in sys.modules


def test_core_dependencies_are_minimal_and_torch_free():
    deps = _pyproject()["project"]["dependencies"]
    joined = " ".join(deps).lower()
    assert "faster-whisper" in joined
    assert "jiwer" in joined
    # torch/whisperx must NOT be core deps — no 3.14 wheels; they live in a venv.
    assert "torch" not in joined
    assert "whisperx" not in joined
