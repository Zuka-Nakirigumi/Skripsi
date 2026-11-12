param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 6)]
    [int]$Section
)

# IMPORTANT:
# Keep these lines at the top of the PowerShell script.
# Python/tqdm may write normal progress information to stderr.
$ErrorActionPreference = "Continue"

if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SectionText = "{0:D2}" -f $Section

$SessionRoot = (
    "outputs\activity_logs\" +
    "pc2_section${SectionText}_$Timestamp"
)

New-Item `
    -ItemType Directory `
    -Force `
    -Path $SessionRoot `
    | Out-Null

$EnvironmentLog = Join-Path `
    $SessionRoot `
    "environment.txt"

$PipFreeze = Join-Path `
    $SessionRoot `
    "pip_freeze.txt"

$GpuCsv = Join-Path `
    $SessionRoot `
    "gpu_usage.csv"

$SectionLog = Join-Path `
    $SessionRoot `
    "section.log"

$SectionReport = Join-Path `
    $SessionRoot `
    "section_report.json"


# ============================================================
# Environment snapshot
# ============================================================

"PRODUCTION SECTION - RTX 3070" |
    Out-File $EnvironmentLog

"Timestamp: $(Get-Date -Format o)" |
    Out-File $EnvironmentLog -Append

"Computer: $env:COMPUTERNAME" |
    Out-File $EnvironmentLog -Append

"Section: $Section" |
    Out-File $EnvironmentLog -Append

"`n--- Python ---" |
    Out-File $EnvironmentLog -Append

python --version 2>&1 |
    Out-File $EnvironmentLog -Append

python -c "import sys; print(sys.executable)" 2>&1 |
    Out-File $EnvironmentLog -Append

"`n--- Git commit ---" |
    Out-File $EnvironmentLog -Append

git rev-parse HEAD 2>&1 |
    Out-File $EnvironmentLog -Append

"`n--- Git status ---" |
    Out-File $EnvironmentLog -Append

git status --short 2>&1 |
    Out-File $EnvironmentLog -Append

"`n--- NVIDIA-SMI ---" |
    Out-File $EnvironmentLog -Append

nvidia-smi 2>&1 |
    Out-File $EnvironmentLog -Append

python -m pip freeze 2>&1 |
    Out-File $PipFreeze


# ============================================================
# GPU monitor
# ============================================================

"timestamp,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_draw_w,power_limit_w" |
    Out-File `
        -FilePath $GpuCsv `
        -Encoding utf8


$GpuMonitor = Start-Job `
    -ArgumentList $GpuCsv `
    -ScriptBlock {

        param($Output)

        while ($true) {

            nvidia-smi `
                -i 0 `
                --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit `
                --format=csv,noheader,nounits `
                | Out-File `
                    -FilePath $Output `
                    -Append `
                    -Encoding utf8

            Start-Sleep `
                -Seconds 10
        }
    }


try {

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "STARTING PC2 PRODUCTION SECTION $Section"
    Write-Host "ACTIVITY LOG: $SessionRoot"
    Write-Host "============================================================"
    Write-Host ""

    & python scripts/production_section.py `
        --machine pc2 `
        --section $Section `
        --device 0 `
        --num-workers 4 `
        --seed 1993 `
        --output-root outputs/experiments `
        --report-path $SectionReport `
        2>&1 |
        Tee-Object `
            -FilePath $SectionLog

    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        throw (
            "PC2 production section $Section " +
            "failed with exit code $ExitCode"
        )
    }

}
finally {

    Stop-Job `
        $GpuMonitor `
        -ErrorAction SilentlyContinue

    Remove-Job `
        $GpuMonitor `
        -Force `
        -ErrorAction SilentlyContinue

    nvidia-smi 2>&1 |
        Out-File (
            Join-Path `
                $SessionRoot `
                "nvidia_smi_final.txt"
        )
}


Write-Host ""
Write-Host "============================================================"
Write-Host "PC2 PRODUCTION SECTION $Section PASSED"
Write-Host "OUTPUT: $SessionRoot"
Write-Host "============================================================"
