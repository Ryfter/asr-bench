# Creates the WhisperX venv that asr-bench auto-detects (./.venv-whisperx).
# WhisperX needs PyTorch, which has no Python 3.14 wheels — so this venv uses 3.12.
# Usage:  ./setup_whisperx_venv.ps1
py -3.12 -m venv .venv-whisperx
.\.venv-whisperx\Scripts\python.exe -m pip install --upgrade pip
.\.venv-whisperx\Scripts\pip.exe install whisperx
Write-Host ""
Write-Host "Done. asr-bench auto-detects .venv-whisperx for the whisperx subprocess adapter."
Write-Host "For diarization: set HF_TOKEN (or pass --hf-token) and accept the gated"
Write-Host "pyannote/speaker-diarization-3.1 model on HuggingFace. Without a token, whisperx"
Write-Host "runs alignment-only."
