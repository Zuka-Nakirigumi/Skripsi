$ErrorActionPreference = "Continue"

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SessionRoot = "outputs\sanity_final\pc2_$Timestamp"
$LogDir = Join-Path $SessionRoot "_activity"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$EnvironmentLog = Join-Path $LogDir "environment.txt"
$GpuCsv = Join-Path $LogDir "gpu_usage.csv"
$TrainingLog = Join-Path $LogDir "sanity_training.log"
$PipFreeze = Join-Path $LogDir "pip_freeze.txt"

"FINAL SANITY - RTX 3070" | Out-File $EnvironmentLog
"Timestamp: $(Get-Date -Format o)" | Out-File $EnvironmentLog -Append
"Computer: $env:COMPUTERNAME" | Out-File $EnvironmentLog -Append
"Python:" | Out-File $EnvironmentLog -Append
python --version 2>&1 | Out-File $EnvironmentLog -Append
"Python executable:" | Out-File $EnvironmentLog -Append
python -c "import sys; print(sys.executable)" 2>&1 | Out-File $EnvironmentLog -Append
"Git commit:" | Out-File $EnvironmentLog -Append
git rev-parse HEAD 2>&1 | Out-File $EnvironmentLog -Append
"Git status:" | Out-File $EnvironmentLog -Append
git status --short 2>&1 | Out-File $EnvironmentLog -Append
"NVIDIA-SMI:" | Out-File $EnvironmentLog -Append
nvidia-smi 2>&1 | Out-File $EnvironmentLog -Append

python -m pip freeze 2>&1 | Out-File $PipFreeze

"timestamp,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_draw_w,power_limit_w" |
    Out-File -FilePath $GpuCsv -Encoding utf8

$GpuMonitor = Start-Job -ArgumentList $GpuCsv -ScriptBlock {
    param($Output)

    while ($true) {
        nvidia-smi `
            --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit `
            --format=csv,noheader,nounits `
            | Out-File -FilePath $Output -Append -Encoding utf8

        Start-Sleep -Seconds 10
    }
}

try {
    & python scripts/sanity_final.py `
        --methods fecam ssre `
        --dataset aid `
        --protocol T5 `
        --backbone r18 `
        --attention none `
        --epochs 1 `
        --max-tasks 2 `
        --device 0 `
        --num-workers 4 `
        --batch fecam=20 `
        --batch ssre=20 `
        --output-root $SessionRoot `
        2>&1 |
        Tee-Object -FilePath $TrainingLog

    $ExitCode = $LASTEXITCODE

    if ($ExitCode -ne 0) {
        throw "PC2 sanity failed with exit code $ExitCode"
    }
}
finally {
    Stop-Job $GpuMonitor -ErrorAction SilentlyContinue
    Remove-Job $GpuMonitor -Force -ErrorAction SilentlyContinue

    nvidia-smi 2>&1 |
        Out-File (Join-Path $LogDir "nvidia_smi_final.txt")
}

Write-Host ""
Write-Host "============================================================"
Write-Host "PC2 FINAL SANITY PASSED"
Write-Host "OUTPUT: $SessionRoot"
Write-Host "============================================================"
