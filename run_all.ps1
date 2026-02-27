# One-click pipeline: train -> evaluate -> export NCNN
# Usage: run at project root
#   powershell -ExecutionPolicy Bypass -File .\run_all.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ========================= User Config (edit here) =========================
# Python executable: leave empty to use current shell environment's python
# Or set an absolute path, e.g. "C:/path/to/python.exe"
$PythonExe = ""

# Dataset root (ImageFolder structure)
# e.g. "dataset" or "D:/datasets/my_cls"
$DataRoot = "dataset01"

# Output directory
$OutDir = "artifacts01"

# Train params
$Train = @{
    ImgSize = 96
    BatchSize = 64
    Epochs = 35
    Lr = 1e-3
    WeightDecay = 2e-4
    NumWorkers = 4
    Seed = 42
    WidthMult = 0.6
    Patience = 8
}

# Eval params
$Eval = @{
    BatchSize = 128
    NumWorkers = 2
    Seed = 42
}

# Whether to export NCNN (calls convert_to_ncnn.ps1)
$RunNcnnExport = $true
# ====================== End User Config (do not edit below) =================

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][scriptblock]$Action
    )

    Write-Host "`n===== $Name =====" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed, exit code: $LASTEXITCODE"
    }
}

function Resolve-PythonExe {
    param([string]$Configured)

    if (-not [string]::IsNullOrWhiteSpace($Configured)) {
        return $Configured
    }

    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython) {
            return $condaPython
        }
    }

    throw 'Python executable not found. Please set $PythonExe in the user config section.'
}

$PythonExe = Resolve-PythonExe -Configured $PythonExe
Write-Host "Using Python: $PythonExe" -ForegroundColor Yellow
& $PythonExe -c "import sys; print('Python executable from runtime:', sys.executable)"

if (-not (Test-Path -LiteralPath $DataRoot)) {
    throw "Data root not found: $DataRoot"
}

if (-not (Test-Path -LiteralPath "./train_tiny_classifier.py")) {
    throw "Missing file: train_tiny_classifier.py"
}
if (-not (Test-Path -LiteralPath "./evaluate_local_accuracy.py")) {
    throw "Missing file: evaluate_local_accuracy.py"
}
if (-not (Test-Path -LiteralPath "./convert_to_ncnn.ps1")) {
    throw "Missing file: convert_to_ncnn.ps1"
}

Invoke-Step -Name "Check Dependencies" -Action {
    & $PythonExe -c "import torch, torchvision, onnx, onnxsim; print('ok')"
    if ($LASTEXITCODE -ne 0) {
        throw "Python environment missing required packages. Install: torch torchvision onnx onnxsim"
    }
}

Invoke-Step -Name "Train Model" -Action {
    $argsTrain = @(
        "./train_tiny_classifier.py",
        "--data-root", $DataRoot,
        "--img-size", "$($Train.ImgSize)",
        "--batch-size", "$($Train.BatchSize)",
        "--epochs", "$($Train.Epochs)",
        "--lr", "$($Train.Lr)",
        "--weight-decay", "$($Train.WeightDecay)",
        "--num-workers", "$($Train.NumWorkers)",
        "--seed", "$($Train.Seed)",
        "--width-mult", "$($Train.WidthMult)",
        "--patience", "$($Train.Patience)",
        "--out-dir", $OutDir
    )
    & $PythonExe @argsTrain
}

$CheckpointPath = Join-Path $OutDir "best_model.pt"
if (-not (Test-Path -LiteralPath $CheckpointPath)) {
    throw "Checkpoint not found after training: $CheckpointPath"
}

Invoke-Step -Name "Evaluate Accuracy" -Action {
    $argsEval = @(
        "./evaluate_local_accuracy.py",
        "--data-root", $DataRoot,
        "--checkpoint", $CheckpointPath,
        "--img-size", "$($Train.ImgSize)",
        "--batch-size", "$($Eval.BatchSize)",
        "--num-workers", "$($Eval.NumWorkers)",
        "--seed", "$($Eval.Seed)"
    )
    & $PythonExe @argsEval
}

if ($RunNcnnExport) {
    Invoke-Step -Name "Export NCNN" -Action {
        $onnxPath = Join-Path $OutDir "tiny_classifier_96.onnx"
        $ncnnParam = Join-Path $OutDir "tiny_classifier_96.ncnn.param"
        $ncnnBin = Join-Path $OutDir "tiny_classifier_96.ncnn.bin"
        $ncnnOptParam = Join-Path $OutDir "tiny_classifier_96.opt.param"
        $ncnnOptBin = Join-Path $OutDir "tiny_classifier_96.opt.bin"

        & "./convert_to_ncnn.ps1" `
            -OnnxPath $onnxPath `
            -OutParam $ncnnParam `
            -OutBin $ncnnBin `
            -OutParamOpt $ncnnOptParam `
            -OutBinOpt $ncnnOptBin
    }
}

Write-Host "`nAll steps finished. Output dir: $OutDir" -ForegroundColor Green
Write-Host ('- Model: {0}' -f (Join-Path $OutDir 'best_model.pt'))
Write-Host ('- Label order: {0}' -f (Join-Path $OutDir 'labels.txt'))
Write-Host ('- Metrics: {0}' -f (Join-Path $OutDir 'metrics.json'))
Write-Host ('- ONNX: {0}' -f (Join-Path $OutDir 'tiny_classifier_96.onnx'))
