@echo off
chcp 65001 >nul
title Emotion Detection AI - Stop Services

set "ROOT_DIR=%~dp0"
set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"

echo ==============================================================================
echo        EMOTION DETECTION AI - STOPPING ALL RUNNING SERVICES
echo ==============================================================================
echo.

if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -c "import subprocess, re; killed = False; (stdout := subprocess.run(['netstat', '-ano'], capture_output=True, text=True).stdout); [print(f'[*] Terminating PID {parts[-1]} on port {port}...') or subprocess.run(['taskkill', '/F', '/PID', parts[-1]], capture_output=True) or globals().update(killed=True) for line in stdout.splitlines() for port in ['8000', '3000'] if f':{port}' in line and 'LISTENING' in line and (parts := line.strip().split())]; print('[OK] All services stopped successfully.' if killed else '[*] No services were running on port 8000 or 3000.')"
) else (
    python -c "import subprocess, re; killed = False; (stdout := subprocess.run(['netstat', '-ano'], capture_output=True, text=True).stdout); [print(f'[*] Terminating PID {parts[-1]} on port {port}...') or subprocess.run(['taskkill', '/F', '/PID', parts[-1]], capture_output=True) or globals().update(killed=True) for line in stdout.splitlines() for port in ['8000', '3000'] if f':{port}' in line and 'LISTENING' in line and (parts := line.strip().split())]; print('[OK] All services stopped successfully.' if killed else '[*] No services were running on port 8000 or 3000.')"
)

echo.
echo Press any key to close this window...
pause >nul
