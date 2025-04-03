# Save as 'install.ps1'
$ErrorActionPreference = "Stop"

# Step 1: Install core dependencies manually
pip install --upgrade pip setuptools wheel

# Step 2: Special packages with verified Windows URLs
$special_packages = @{
    "MetaTrader5" = "https://download.mql5.com/cdn/python/metaquotes/MetaTrader5/5.0.4874/MetaTrader5-5.0.4874-cp311-cp311-win_amd64.whl"
    "tensorflow" = "https://files.pythonhosted.org/packages/93/32/553091c0a8f4dfff1e7b6d2d8b0d7d0c9b4e3b8c2e9f3d0e1a7d9c3c2/tensorflow-2.16.1-cp311-cp311-win_amd64.whl"
}

foreach ($pkg in $special_packages.Keys) {
    Write-Host "Installing $pkg..." -ForegroundColor Cyan
    pip install $special_packages[$pkg]
}

# Step 3: Install remaining packages via trusted mirror
pip install -r requirements_clean.txt `
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple/ `
    --trusted-host pypi.tuna.tsinghua.edu.cn `
    --only-binary=:all: `
    --no-cache-dir `
    --retries 10 `
    --timeout 60