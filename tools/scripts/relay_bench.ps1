[CmdletBinding()]
param(
    [string]$Model = (Join-Path $PSScriptRoot '..\models\mamba_flow.safetensors')
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$build = if ($env:FLOWEDGE_BUILD_DIR) { $env:FLOWEDGE_BUILD_DIR } else { Join-Path $root 'build-relay' }
$iterations = if ($env:FLOWEDGE_RELAY_BENCH_ITERS) { $env:FLOWEDGE_RELAY_BENCH_ITERS } else { 500 }
$buildJobs = 2
if ($env:FLOWEDGE_BUILD_JOBS) { $buildJobs = [int]$env:FLOWEDGE_BUILD_JOBS }
if ($buildJobs -lt 1) { throw "FLOWEDGE_BUILD_JOBS must be positive" }

if (-not (Test-Path -LiteralPath $Model)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Model) | Out-Null
    Invoke-WebRequest 'https://huggingface.co/ReForceMind/mamba_flow/resolve/main/mamba_flow.safetensors' -OutFile $Model
}

$generator = if (Get-Command ninja -ErrorAction SilentlyContinue) { 'Ninja' } else { 'Visual Studio 17 2022' }
cmake -S $root -B $build -G $generator -DFLOWEDGE_RELAY=ON -DFLOWEDGE_BENCH=ON -DFLOWEDGE_TESTS=OFF
cmake --build $build --config Release --parallel $buildJobs

$report = if ($env:FLOWEDGE_RELAY_BENCH_REPORT_DIR) {
    New-Item -ItemType Directory -Force -Path $env:FLOWEDGE_RELAY_BENCH_REPORT_DIR | Out-Null
    Join-Path $env:FLOWEDGE_RELAY_BENCH_REPORT_DIR ("relay-bench-{0:yyyyMMddTHHmmssZ}.txt" -f (Get-Date).ToUniversalTime())
}
if ($report) {
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    @("system=$($os.Caption) $($os.Version)", "cpu=$($cpu.Name)", "commit=$(git -C $root rev-parse --short HEAD)") | Set-Content $report -Encoding utf8
}

function Resolve-Bench([string]$name) {
    foreach ($candidate in @((Join-Path $build "$name.exe"), (Join-Path $build "Release\$name.exe"))) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "Missing benchmark executable: $name"
}

function Invoke-Bench([string]$name, [string[]]$arguments) {
    $exe = Resolve-Bench $name
    if ($report) {
        "command=$exe $($arguments -join ' ')" | Add-Content $report -Encoding utf8
        $lines = & $exe @arguments 2>&1
        $lines | ForEach-Object { "$_" | Add-Content $report -Encoding utf8; $_ }
    } else { & $exe @arguments }
    if ($LASTEXITCODE) { throw "$name failed with exit code $LASTEXITCODE" }
}

Invoke-Bench flowedge_relay_bench @($Model, $iterations, $(if ($env:FLOWEDGE_THREADS) { $env:FLOWEDGE_THREADS } else { 0 }))
Invoke-Bench flowedge_relay_pool_bench @($Model, $iterations, $(if ($env:FLOWEDGE_RELAY_POOL_WORKERS) { $env:FLOWEDGE_RELAY_POOL_WORKERS } else { 2 }), $(if ($env:FLOWEDGE_RELAY_POOL_THREADS) { $env:FLOWEDGE_RELAY_POOL_THREADS } else { 0 }))
Invoke-Bench flowedge_cooperative_job_bench @($(if ($env:FLOWEDGE_RELAY_COOPERATIVE_BENCH_ITERS) { $env:FLOWEDGE_RELAY_COOPERATIVE_BENCH_ITERS } else { 1000000 }))
Invoke-Bench flowedge_job_queue_bench @($(if ($env:FLOWEDGE_RELAY_QUEUE_BENCH_ROUNDS) { $env:FLOWEDGE_RELAY_QUEUE_BENCH_ROUNDS } else { 3000 }))
Invoke-Bench flowedge_job_qos_bench @($(if ($env:FLOWEDGE_RELAY_QOS_BENCH_ITERS) { $env:FLOWEDGE_RELAY_QOS_BENCH_ITERS } else { 1000000 }))
Invoke-Bench flowedge_mamba_stream_bench @($Model, $iterations)
Invoke-Bench flowedge_worker_drain_bench @($(if ($env:FLOWEDGE_RELAY_DRAIN_BENCH_ITERS) { $env:FLOWEDGE_RELAY_DRAIN_BENCH_ITERS } else { 10000 }))
Invoke-Bench flowedge_action_delivery_bench @($(if ($env:FLOWEDGE_RELAY_ACTION_BENCH_ITERS) { $env:FLOWEDGE_RELAY_ACTION_BENCH_ITERS } else { 1000000 }))

if ($report) { Write-Output "wrote $report" }
