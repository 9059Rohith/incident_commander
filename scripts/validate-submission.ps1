param(
    [Parameter(Mandatory = $true)]
    [string]$PingUrl,

    [string]$RepoDir = "."
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $Message"
}

function Fail([string]$Message) {
    Write-Host "FAILED -- $Message" -ForegroundColor Red
    exit 1
}

function Pass([string]$Message) {
    Write-Host "PASSED -- $Message" -ForegroundColor Green
}

$repoPath = Resolve-Path $RepoDir
$pingBase = $PingUrl.TrimEnd('/')

Write-Host ""
Write-Host "========================================"
Write-Host "  OpenEnv Submission Validator (Windows)"
Write-Host "========================================"
Write-Step "Repo:     $repoPath"
Write-Step "Ping URL: $pingBase"
Write-Host ""

Write-Step "Step 1/3: Pinging HF Space ($pingBase/reset) ..."
try {
    $response = Invoke-RestMethod -Uri "$pingBase/reset" -Method Post -ContentType "application/json" -Body "{}" -TimeoutSec 30
    Pass "HF Space is live and responds to /reset"
}
catch {
    if ($_.Exception.Response) {
        $status = [int]$_.Exception.Response.StatusCode
        Fail "HF Space /reset returned HTTP $status (expected 200)"
    }
    Fail "HF Space not reachable (connection failed or timed out)"
}

Write-Step "Step 2/3: Running docker build ..."
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    Fail "docker command not found"
}

$dockerContext = $null
if (Test-Path (Join-Path $repoPath "Dockerfile")) {
    $dockerContext = $repoPath
}
elseif (Test-Path (Join-Path $repoPath "server/Dockerfile")) {
    $dockerContext = Join-Path $repoPath "server"
}
else {
    Fail "No Dockerfile found in repo root or server/ directory"
}

Write-Step "Found Dockerfile in $dockerContext"
& docker build $dockerContext
if ($LASTEXITCODE -ne 0) {
    Fail "Docker build failed"
}
Pass "Docker build succeeded"

Write-Step "Step 3/3: Running OpenEnv validate ..."
Push-Location $repoPath
try {
    $workspaceParent = Split-Path $repoPath -Parent
    $venvCandidates = @(
        (Join-Path $repoPath ".venv/Scripts/python.exe"),
        (Join-Path $workspaceParent ".venv/Scripts/python.exe")
    )
    $venvPython = $venvCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($venvPython) {
        & $venvPython -m openenv.cli validate
    }
    elseif (Get-Command openenv -ErrorAction SilentlyContinue) {
        & openenv validate
    }
    else {
        & python -m openenv.cli validate
    }
    if ($LASTEXITCODE -ne 0) {
        Fail "openenv validate failed"
    }
    Pass "openenv validate passed"
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "========================================"
Write-Host "  All 3/3 checks passed!" -ForegroundColor Green
Write-Host "  Your submission is ready to submit." -ForegroundColor Green
Write-Host "========================================"
Write-Host ""