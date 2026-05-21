param(
    [string]$UnityPath = "C:\Program Files\Unity\Hub\Editor\6000.3.11f1\Editor\Unity.exe",
    [string]$RunId = "unity_procedural_001",
    [int]$Episodes = 4,
    [int]$FramesPerEpisode = 180,
    [int]$CaptureSize = 128,
    [double]$HistorySec = 1.0,
    [double]$FutureSec = 3.0,
    [double]$SampleFps = 5.0,
    [int]$Stride = 2,
    [int]$UnityTimeoutSec = 900,
    [switch]$SkipBuildSamples,
    [switch]$EnablePackageManager
)

$ErrorActionPreference = "Stop"

function Resolve-RepoPath([string]$RelativePath) {
    return [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $RelativePath))
}

function ConvertTo-ProcessArgument([string]$Value) {
    if ($null -eq $Value) {
        return '""'
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-UnityBatch([string[]]$UnityArgs, [int]$TimeoutSec) {
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $UnityPath
    $startInfo.WorkingDirectory = $RepoRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.Arguments = ($UnityArgs | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " "

    $process = [System.Diagnostics.Process]::Start($startInfo)
    if (-not $process.WaitForExit($TimeoutSec * 1000)) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        Get-Process Unity.Licensing.Client -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        throw "Unity batch generation timed out after $TimeoutSec seconds. Check $LogPath."
    }

    if ($process.ExitCode -ne 0) {
        if (Test-Path $LogPath) {
            Get-Content $LogPath -Tail 120
        }
        throw "Unity batch generation failed with exit code $($process.ExitCode). Check $LogPath."
    }
}

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$ProjectPath = Resolve-RepoPath "client\unity_path_client"
$RawRoot = Resolve-RepoPath "data\wit_vz\raw"
$RawRunPath = Join-Path $RawRoot $RunId
$ProcessedRunPath = Resolve-RepoPath ("data\wit_vz\processed\" + $RunId)
$LogPath = Resolve-RepoPath ".tmp_unity_dataset_generation.log"

if (-not (Test-Path $UnityPath)) {
    throw "Unity executable was not found: $UnityPath"
}

Remove-Item $LogPath -ErrorAction SilentlyContinue

$unityArgs = @(
    "-batchmode",
    "-quit",
    "-projectPath",
    $ProjectPath,
    "-executeMethod",
    "DDPDUnityDatasetBatch.GenerateDefaultAndQuit",
    "-logFile",
    $LogPath,
    "--ddpd-run-id",
    $RunId,
    "--ddpd-raw-root",
    $RawRoot,
    "--ddpd-episodes",
    [string]$Episodes,
    "--ddpd-frames-per-episode",
    [string]$FramesPerEpisode,
    "--ddpd-capture-size",
    [string]$CaptureSize
)

if (-not $EnablePackageManager) {
    $unityArgs = @("-noUpm") + $unityArgs
}

Write-Host "Generating Unity raw dataset: $RawRunPath"
Invoke-UnityBatch -UnityArgs $unityArgs -TimeoutSec $UnityTimeoutSec

$manifestPath = Join-Path $RawRunPath "manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "Unity completed, but raw manifest was not found: $manifestPath"
}

if (-not $SkipBuildSamples) {
    Write-Host "Building processed samples: $ProcessedRunPath"
    & uv run python -m src.wit_vz.build_samples `
        --raw $RawRunPath `
        --out $ProcessedRunPath `
        --history-sec $HistorySec `
        --future-sec $FutureSec `
        --sample-fps $SampleFps `
        --stride $Stride `
        --split episode

    if ($LASTEXITCODE -ne 0) {
        throw "build_samples failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Unity raw dataset ready: $RawRunPath"
if (-not $SkipBuildSamples) {
    Write-Host "Processed samples ready: $ProcessedRunPath"
}
