$environments = @("env", "gg_env", "goldengoose_env", "myenv", "trading_env", "venv")

foreach ($env in $environments) {
    $env_path = "C:\GoldenGoose\$env"
    
    Write-Host "`n=== Checking $env_path ===" -ForegroundColor Cyan
    
    if (-Not (Test-Path "$env_path\Scripts\activate")) {
        Write-Host "Invalid environment - missing activation scripts" -ForegroundColor Red
        continue
    }

    try {
        # Activate environment
        & "$env_path\Scripts\activate.ps1"
        
        # 1. Check package consistency
        pip list --format=freeze | Out-File "$env_path-packages.txt"
        $diff = Compare-Object (Get-Content requirements.txt) (Get-Content "$env_path-packages.txt")
        
        if ($diff) {
            Write-Host "Package mismatch detected!" -ForegroundColor Red
            $diff | Where-Object { $_.SideIndicator -eq "=>" } | ForEach-Object {
                Write-Host "Extra package: $($_.InputObject)" -ForegroundColor Yellow
            }
            $diff | Where-Object { $_.SideIndicator -eq "<=" } | ForEach-Object {
                Write-Host "Missing package: $($_.InputObject)" -ForegroundColor Red
            }
        } else {
            Write-Host "Packages match requirements.txt" -ForegroundColor Green
        }

        # 2. Verify Python version
        $python_version = & "python" --version 2>&1
        Write-Host "Python Version: $python_version"

        # 3. Corruption check
        Write-Host "`nRunning integrity checks..." -ForegroundColor White
        try {
            python -c "import tensorflow as tf; print('TensorFlow OK:', tf.__version__)"
            python -c "import MetaTrader5 as mt5; print('MetaTrader5 OK:', mt5.__version__)"
            $pip_check = pip check 2>&1
            if ($pip_check -match "broken") {
                Write-Host "Broken dependencies found!" -ForegroundColor Red
                $pip_check
            } else {
                Write-Host "Environment integrity verified" -ForegroundColor Green
            }
        } catch {
            Write-Host "Critical error during imports: $_" -ForegroundColor Red
        }

    } finally {
        deactivate
    }
}

Write-Host "`n=== Validation Complete ===" -ForegroundColor Magenta
Write-Host "Recommended environments to KEEP (all checks passed):" -ForegroundColor Green
Write-Host "Recommended environments to DELETE:" -ForegroundColor Red