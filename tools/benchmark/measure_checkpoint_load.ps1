param(
    [Parameter(Mandatory = $true)]
    [string]$Baseline,
    [Parameter(Mandatory = $true)]
    [string]$Candidate,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [ValidateRange(1, 100)]
    [int]$Trials = 5,
    [ValidateRange(0, 20)]
    [int]$Warmup = 1
)

$ErrorActionPreference = "Stop"

$baselinePath = (Resolve-Path -LiteralPath $Baseline).Path
$candidatePath = (Resolve-Path -LiteralPath $Candidate).Path
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$outputPath = [System.IO.Path]::GetFullPath($Output)
$outputDirectory = Split-Path -Parent $outputPath
if ($outputDirectory -and -not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

function Measure-CheckpointLoad([string]$Name, [string]$Executable, [int]$Trial) {
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Executable
    $startInfo.Arguments = '"' + $checkpointPath.Replace('"', '\"') + '" --json'
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    try {
        $watch = [System.Diagnostics.Stopwatch]::StartNew()
        if (-not $process.Start()) {
            throw "failed to start $Executable"
        }
        $peakPrivateBytes = 0L
        $peakWorkingSetBytes = 0L
        while (-not $process.HasExited) {
            try {
                $process.Refresh()
                $peakPrivateBytes = [Math]::Max(
                    $peakPrivateBytes, $process.PrivateMemorySize64)
                $peakWorkingSetBytes = [Math]::Max(
                    $peakWorkingSetBytes, $process.WorkingSet64)
            }
            catch {
                # The process can exit between HasExited and Refresh.
            }
            Start-Sleep -Milliseconds 2
        }
        $process.WaitForExit()
        $process.Refresh()
        $watch.Stop()
        $peakWorkingSetBytes = [Math]::Max(
            $peakWorkingSetBytes, $process.PeakWorkingSet64)
        $standardOutput = $process.StandardOutput.ReadToEnd()
        $errorText = $process.StandardError.ReadToEnd()
        if ($process.ExitCode -ne 0) {
            throw "$Name checkpoint load failed with exit code $($process.ExitCode): $errorText"
        }
        if (-not $standardOutput.TrimStart().StartsWith("{")) {
            throw "$Name checkpoint load did not emit JSON"
        }
        return [ordered]@{
            name = $Name
            trial = $Trial
            wall_ms = [Math]::Round($watch.Elapsed.TotalMilliseconds, 3)
            peak_private_bytes = $peakPrivateBytes
            peak_working_set_bytes = $peakWorkingSetBytes
        }
    }
    finally {
        $process.Dispose()
    }
}

for ($index = 0; $index -lt $Warmup; ++$index) {
    Measure-CheckpointLoad "baseline_warmup" $baselinePath $index | Out-Null
    Measure-CheckpointLoad "candidate_warmup" $candidatePath $index | Out-Null
}

$samples = @()
for ($trial = 1; $trial -le $Trials; ++$trial) {
    if (($trial % 2) -eq 1) {
        $samples += Measure-CheckpointLoad "baseline" $baselinePath $trial
        $samples += Measure-CheckpointLoad "candidate" $candidatePath $trial
    }
    else {
        $samples += Measure-CheckpointLoad "candidate" $candidatePath $trial
        $samples += Measure-CheckpointLoad "baseline" $baselinePath $trial
    }
}

function Get-Median($Values) {
    $sorted = @($Values | Sort-Object)
    return [double]$sorted[[Math]::Floor($sorted.Count / 2)]
}

$baselineSamples = @($samples | Where-Object { $_.name -eq "baseline" })
$candidateSamples = @($samples | Where-Object { $_.name -eq "candidate" })
$baselineWall = Get-Median ($baselineSamples | ForEach-Object { $_.wall_ms })
$candidateWall = Get-Median ($candidateSamples | ForEach-Object { $_.wall_ms })
$baselinePrivate = Get-Median ($baselineSamples | ForEach-Object { $_.peak_private_bytes })
$candidatePrivate = Get-Median ($candidateSamples | ForEach-Object { $_.peak_private_bytes })
$baselineWorkingSet = Get-Median (
    $baselineSamples | ForEach-Object { $_.peak_working_set_bytes })
$candidateWorkingSet = Get-Median (
    $candidateSamples | ForEach-Object { $_.peak_working_set_bytes })

$report = [ordered]@{
    schema_version = 1
    scope = "flowedge-inspect process load; includes checkpoint validation, fingerprint, and runtime construction"
    host = [ordered]@{
        os = [System.Environment]::OSVersion.VersionString
        processor = $env:PROCESSOR_IDENTIFIER
    }
    checkpoint = [ordered]@{
        file = [System.IO.Path]::GetFileName($checkpointPath)
        bytes = (Get-Item -LiteralPath $checkpointPath).Length
        sha256 = (Get-FileHash -LiteralPath $checkpointPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    executables = [ordered]@{
        baseline = [ordered]@{
            file = [System.IO.Path]::GetFileName($baselinePath)
            sha256 = (Get-FileHash -LiteralPath $baselinePath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        candidate = [ordered]@{
            file = [System.IO.Path]::GetFileName($candidatePath)
            sha256 = (Get-FileHash -LiteralPath $candidatePath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    warmup = $Warmup
    trials = $Trials
    summary = [ordered]@{
        baseline_median_wall_ms = $baselineWall
        candidate_median_wall_ms = $candidateWall
        wall_reduction_fraction = 1.0 - ($candidateWall / $baselineWall)
        baseline_median_peak_private_bytes = $baselinePrivate
        candidate_median_peak_private_bytes = $candidatePrivate
        private_bytes_removed = $baselinePrivate - $candidatePrivate
        baseline_median_peak_working_set_bytes = $baselineWorkingSet
        candidate_median_peak_working_set_bytes = $candidateWorkingSet
        working_set_bytes_removed = $baselineWorkingSet - $candidateWorkingSet
    }
    samples = $samples
}

$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $outputPath -Encoding utf8
Write-Output $outputPath
