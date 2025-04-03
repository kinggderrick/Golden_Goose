@echo off
:: ==============================================
:: 4-Core Optimized Trading Bot Launcher
:: ==============================================

rem 1. Clean previous sessions
taskkill /IM python.exe /F > nul 2>&1

rem 2. Set core affinity (physical cores 0-3)
set /A CORE_MASK=0x0000000F
start /B /AFF %CORE_MASK% /HIGH cmd /c "call .\venv\Scripts\activate"

rem 3. GPU Optimization (If available)
set TF_ENABLE_ONEDNN_OPTS=1
set TF_CPP_MIN_LOG_LEVEL=3

rem 4. Python Execution with 4-Core Settings
python -B -OO bot.py ^
  --prefetch 256 ^  :: Reduced from 512 for 4-core
  --batch 64 ^      :: Smaller batches for L3 cache
  --disable-debug

rem 5. Error Handling
if %errorlevel% neq 0 (
   echo [%time%] RESTARTING...
   timeout /t 10 /nobreak
   start "" /NORMAL "%~dpnx0"
)

pause