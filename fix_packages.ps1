# 1. Create fresh environment
python -m venv goldengoose_prod

# 2. Activate environment
.\goldengoose_prod\Scripts\activate.ps1

# 3. Update core packages first
python -m pip install --upgrade pip setuptools wheel --trusted-host pypi.tuna.tsinghua.edu.cn --retries 10 --timeout 60

# 4. Install requirements with multiple fallback mirrors
$mirrors = @(
    "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "https://mirrors.aliyun.com/pypi/simple/",
    "https://pypi.org/simple"
)

foreach ($mirror in $mirrors) {
    try {
        python -m pip install --force-reinstall -r requirements.txt `
            --only-binary=:all: `
            --trusted-host $mirror.Split('/')[2] `
            --index-url $mirror `
            --retries 5 `
            --timeout 30 `
            --progress-bar on
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Successfully installed packages using $mirror" -ForegroundColor Green
            break
        }
    } catch {
        Write-Host "Failed with $mirror - trying next mirror..." -ForegroundColor Yellow
    }
}

# 5. Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} OK')"
python -c "import MetaTrader5 as mt5; print(f'MetaTrader5 {mt5.__version__} OK')"

# 6. Cleanup old environments
Get-ChildItem -Directory -Filter "*env*" -Exclude "goldengoose_prod" | Remove-Item -Recurse -Force