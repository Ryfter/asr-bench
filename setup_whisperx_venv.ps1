# Creates the WhisperX venv that asr-bench auto-detects (./.venv-whisperx).
# WhisperX needs PyTorch, which has no Python 3.14 wheels — so this venv uses 3.12.
#
# Usage:  ./setup_whisperx_venv.ps1                 # default CUDA build (cu128)
#         ./setup_whisperx_venv.ps1 -CudaIndex cpu  # CPU-only torch
#         ./setup_whisperx_venv.ps1 -CudaIndex cu124 # older CUDA toolkit
#
# Why -CudaIndex: `pip install whisperx` pulls PyTorch from PyPI, which on Windows
# is the CPU-ONLY wheel — torch.cuda.is_available() comes back False and every run
# silently falls back to CPU. We install the CUDA build from PyTorch's own index
# FIRST so whisperx then sees torch already satisfied. cu128 has Blackwell (RTX
# 50xx / sm_120) support; use cu124 for older toolkits, or cpu to skip CUDA.
param(
    [string]$CudaIndex = "cu128"
)

py -3.12 -m venv .venv-whisperx
.\.venv-whisperx\Scripts\python.exe -m pip install --upgrade pip

if ($CudaIndex -eq "cpu") {
    .\.venv-whisperx\Scripts\pip.exe install torch torchvision torchaudio
} else {
    .\.venv-whisperx\Scripts\pip.exe install torch torchvision torchaudio `
        --index-url "https://download.pytorch.org/whl/$CudaIndex"
}
.\.venv-whisperx\Scripts\pip.exe install whisperx

Write-Host ""
Write-Host "Verifying CUDA in the venv..."
.\.venv-whisperx\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'))"

Write-Host ""
Write-Host "Done. asr-bench auto-detects .venv-whisperx for the whisperx subprocess adapter."
Write-Host "For diarization: set HF_TOKEN (or pass --hf-token) and accept the gated"
Write-Host "pyannote/speaker-diarization-community-1 model on HuggingFace (pyannote-audio 4.x"
Write-Host "unified on this single self-contained repo). Without a token, whisperx runs"
Write-Host "alignment-only. For long recordings, pass --min-speakers/--max-speakers to keep"
Write-Host "pyannote from over-clustering."
