# Local verification for Grok Bot (Windows).
# C++ tests plus the Python parity slice of scripts/verify_all.sh against a Release tree.
param(
    [string]$BuildDir = $(if ($env:FLOWEDGE_BUILD_DIR) { $env:FLOWEDGE_BUILD_DIR } else { "build-win-clang" }),
    [string]$Model = "models/mamba_flow.safetensors"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Build = Join-Path $Root $BuildDir
if (-not (Test-Path $Build)) {
    throw "build dir not found: $Build (configure FLOWEDGE_TESTS=ON FLOWEDGE_PYTHON=ON and build Release first)"
}

function Test-PythonMods([string]$Exe, [string]$Code) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Exe -c $Code 1>$null 2>$null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    return $ok
}

function Resolve-VerifyPython {
    $candidates = @()
    if ($env:PYTHON) { $candidates += $env:PYTHON }
    $candidates += (Join-Path $Root "venv\Scripts\python.exe")
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { $candidates += $cmd.Source }
    $candidates += @(
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
    )
    $seen = @{}
    $fallback = $null
    foreach ($exe in $candidates) {
        if (-not $exe -or $seen.ContainsKey($exe) -or -not (Test-Path $exe)) { continue }
        $seen[$exe] = $true
        if (-not (Test-PythonMods $exe "import sys")) { continue }
        if (-not $fallback) { $fallback = $exe }
        if (Test-PythonMods $exe "import numpy, safetensors") { return $exe }
    }
    return $fallback
}

$env:PYTHONPATH = "$Build;$(Join-Path $Root 'integrations\lerobot\src')"

Write-Host "ctest $Build"
ctest --test-dir $Build -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Tests = Get-ChildItem -Path $Build -Filter "flowedge_tests.exe" | Select-Object -First 1
if ($Tests) {
    & $Tests.FullName --gtest_filter="Matmul*:Conv*:DenseConv1d*:DiffusionConvolution*:Mish*:GroupNorm*:Gelu*:LayerNorm*:Softmax*:CachedCausalAttention*:Transformer.*"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$Python = Resolve-VerifyPython
if (-not $Python) {
    Write-Host "note: python unavailable; C++ tests passed"
    Write-Host "FlowEdge local verification passed (C++ only)"
    exit 0
}
Write-Host "python $Python"

& $Python -m unittest discover -s integrations/lerobot/tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ModelPath = Join-Path $Root $Model
if (-not (Test-Path $ModelPath)) {
    Write-Host "note: $Model missing; skipped ULP and python parity"
    Write-Host "FlowEdge local verification passed (C++ + adapter tests)"
    exit 0
}

$HasNumpySafetensors = Test-PythonMods $Python "import numpy, safetensors"
$HasTorch = Test-PythonMods $Python "import torch, numpy, safetensors"
if ($HasNumpySafetensors) {
    & $Python tools/benchmark/compare_native_diffusion.py --help
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m flowedge_dev verify delivery smoke --output "$Build/delivery-contract-smoke.json"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m flowedge_dev verify head $BuildDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "note: numpy+safetensors verification dependencies unavailable"
}
if ($HasTorch) {
    & $Python -m flowedge_dev verify diffusion $BuildDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python -m flowedge_dev verify ulp $ModelPath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "note: PyTorch ULP/diffusion dependencies unavailable"
}

$TinySrc = Join-Path $Root "models\tiny-gpt2"
$TinyConv = Join-Path $Root "models\tiny-gpt2.flowedge.safetensors"
$HasTransformers = Test-PythonMods $Python "import torch, transformers"
if ($HasTransformers -and (Test-Path $TinySrc) -and (Test-Path $TinyConv)) {
    $Forward = Get-ChildItem -Path $Build -Filter "transformer_forward.exe" | Select-Object -First 1
    $Latency = Get-ChildItem -Path $Build -Filter "transformer_latency.exe" | Select-Object -First 1
    if ($Forward) {
        & $Python -m flowedge_dev verify transformer $TinySrc $TinyConv --binary $Forward.FullName --model-id sshleifer/tiny-gpt2
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    if ($Latency) {
        & $Latency.FullName $TinyConv --threads 0 --warmup 2 --iters 8 1 2 3 4
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
} else {
    Write-Host "note: tiny-gpt2 or transformers unavailable; skipped Transformer HF parity"
}

Write-Host "FlowEdge local verification passed"
