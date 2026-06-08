# Creates the NeMo venv that asr-bench auto-detects (./.venv-nemo).
# NVIDIA NeMo needs PyTorch, which has no Python 3.14 wheels -- so this venv uses
# 3.12. NeMo also has aggressive dependency pins (numpy>=2.0, transformers,
# lightning) that can collide with WhisperX/pyannote, so it gets its OWN venv --
# do NOT share with .venv-whisperx.
#
# Usage:  ./setup_nemo_venv.ps1                 # default CUDA build (cu128)
#         ./setup_nemo_venv.ps1 -CudaIndex cpu  # CPU-only torch (no GPU bench)
#         ./setup_nemo_venv.ps1 -CudaIndex cu124 # older CUDA toolkit
#
# Why -CudaIndex: a bare `pip install nemo_toolkit[asr]` pulls PyTorch from PyPI,
# which on Windows is the CPU-ONLY wheel -- torch.cuda.is_available() comes back
# False and every run silently falls back to CPU. We install the CUDA build from
# PyTorch's own index FIRST so NeMo then sees torch already satisfied. cu128 has
# Blackwell (RTX 50xx / sm_120) support; use cu124 for older toolkits, or cpu to
# skip CUDA.
param(
    [string]$CudaIndex = "cu128"
)

py -3.12 -m venv .venv-nemo
.\.venv-nemo\Scripts\python.exe -m pip install --upgrade pip

# 1) torch FIRST (CUDA build) so NeMo doesn't pull the CPU-only wheel.
if ($CudaIndex -eq "cpu") {
    .\.venv-nemo\Scripts\pip.exe install torch torchvision torchaudio
} else {
    .\.venv-nemo\Scripts\pip.exe install torch torchvision torchaudio `
        --index-url "https://download.pytorch.org/whl/$CudaIndex"
}

# 2) NeMo (ASR collection). Pure-Python wheel; all version pressure is from torch.
.\.venv-nemo\Scripts\pip.exe install "nemo_toolkit[asr]"

Write-Host ""
Write-Host "Verifying CUDA in the venv..."
.\.venv-nemo\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'))"
if ($CudaIndex -ne "cpu") {
    .\.venv-nemo\Scripts\python.exe -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: torch.cuda.is_available() is False -- the CPU-only wheel got installed." -ForegroundColor Red
        Write-Host "Delete .venv-nemo and re-run; ensure the cu128 index URL was used." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Recording the installed version triple (record this in CLAUDE.md/SPEC.md)..."
.\.venv-nemo\Scripts\python.exe -c "import torch, nemo; print('torch', torch.__version__, '| nemo', nemo.__version__)"

Write-Host ""
Write-Host "Done. asr-bench auto-detects .venv-nemo for the nemo subprocess adapter."
Write-Host "Next: smoke-test the runner on one short clip before a full benchmark:"
Write-Host "  .\.venv-nemo\Scripts\python.exe nemo_runner.py --audio SHORT.wav --model nvidia/parakeet-tdt-0.6b-v2 --device cuda"
Write-Host "  .\.venv-nemo\Scripts\python.exe nemo_runner.py --audio SHORT.wav --model nvidia/canary-qwen-2.5b --device cuda"
Write-Host "Each must print a single JSON document on stdout (Parakeet has segments/words;"
Write-Host "Canary is text-only). Then run a full benchmark:"
Write-Host "  python asr_bench.py --models large-v3-turbo,parakeet-tdt-0.6b-v2,canary-qwen-2.5b --device cuda"
