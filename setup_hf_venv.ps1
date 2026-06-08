# Creates the HF-transformers venv that asr-bench auto-detects (./.venv-hf).
# wav2vec2 / Conformer need PyTorch + transformers, which have no Python 3.14
# wheels -- so this venv uses 3.12. Kept separate from .venv-nemo / .venv-whisperx
# to avoid dependency-pin collisions.
#
# Usage:  ./setup_hf_venv.ps1                 # default CUDA build (cu128)
#         ./setup_hf_venv.ps1 -CudaIndex cpu  # CPU-only torch (no GPU bench)
#         ./setup_hf_venv.ps1 -CudaIndex cu124 # older CUDA toolkit
#
# Why -CudaIndex: a bare `pip install torch` on Windows pulls the CPU-only wheel,
# so torch.cuda.is_available() comes back False and every run silently falls back
# to CPU. We install the CUDA build from PyTorch's own index FIRST. cu128 has
# Blackwell (RTX 50xx / sm_120) support; use cu124 for older toolkits, or cpu.
param(
    [string]$CudaIndex = "cu128"
)

py -3.12 -m venv .venv-hf
.\.venv-hf\Scripts\python.exe -m pip install --upgrade pip

if ($CudaIndex -eq "cpu") {
    .\.venv-hf\Scripts\pip.exe install torch torchvision torchaudio
} else {
    .\.venv-hf\Scripts\pip.exe install torch torchvision torchaudio `
        --index-url "https://download.pytorch.org/whl/$CudaIndex"
}
.\.venv-hf\Scripts\pip.exe install "transformers>=4.40" "soundfile"

Write-Host ""
Write-Host "Verifying CUDA in the venv..."
.\.venv-hf\Scripts\python.exe -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'))"
if ($CudaIndex -ne "cpu") {
    .\.venv-hf\Scripts\python.exe -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "ERROR: torch.cuda.is_available() is False -- the CPU-only wheel got installed." -ForegroundColor Red
        Write-Host "Delete .venv-hf and re-run; ensure the cu128 index URL was used." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Recording the installed version triple (record this in CLAUDE.md/SPEC.md)..."
.\.venv-hf\Scripts\python.exe -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

Write-Host ""
Write-Host "Done. asr-bench auto-detects .venv-hf for the hf subprocess adapter."
Write-Host "Smoke-test the runner on one short clip before a full benchmark:"
Write-Host "  .\.venv-hf\Scripts\python.exe hf_runner.py --audio SHORT.wav --model facebook/wav2vec2-large-960h --device cuda"
Write-Host "Then a full benchmark:"
Write-Host "  python asr_bench.py --models large-v3-turbo,wav2vec2-large-960h,wav2vec2-conformer-large --device cuda"
