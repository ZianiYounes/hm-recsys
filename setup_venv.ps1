# =====================================================================
# Creation du venv + PyTorch nightly CUDA 12.8 (RTX 5080 / Blackwell)
# Usage (PowerShell, a la racine du projet) :
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\setup_venv.ps1
# Prerequis : Python 3.11+ (py -3.11), pilote NVIDIA recent (570+).
# Pas besoin d'installer le CUDA Toolkit : les wheels embarquent le runtime CUDA.
# =====================================================================

$ErrorActionPreference = "Stop"

Write-Host "1/6 Creation du venv (.venv) avec Python 3.11" -ForegroundColor Cyan
py -3.11 -m venv .venv

Write-Host "2/6 Activation" -ForegroundColor Cyan
. .\.venv\Scripts\Activate.ps1

Write-Host "3/6 Mise a jour de pip / setuptools / wheel" -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "4/6 PyTorch nightly cu128 (torch + torchvision)" -ForegroundColor Cyan
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

Write-Host "5/6 Dependances du projet" -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "6/6 Verification GPU" -ForegroundColor Cyan
python scripts\check_gpu.py

Write-Host "Termine. Pour reactiver plus tard : .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
