#!/usr/bin/env bash
# setup_wtm_venv.sh — Apple Silicon venv for whisper-turbo-mlx (wtm)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${ROOT}/.venv-wtm"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "WARNING: MLX Whisper Turbo targets Apple Silicon. Continuing anyway." >&2
fi

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python3 not found" >&2
  exit 1
fi

"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip
pip install whisper-turbo-mlx

python - <<'PY'
try:
    import mlx.core  # noqa: F401
    from whisper_turbo import transcribe  # noqa: F401
    print("OK: whisper-turbo-mlx + MLX importable")
except Exception as exc:
    raise SystemExit(f"verify failed: {exc}")
PY

echo "Created ${VENV}. Run:"
echo "  python asr_bench.py --models large-v3-turbo-mlx --corpus ./test-corpus --limit 1"
