@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Emotion Detection AI - Production Runner

set "ROOT_DIR=%~dp0"
set "PYTHON_EXE=%ROOT_DIR%.venv\Scripts\python.exe"
set "API_DIR=%ROOT_DIR%apps\api"
set "WEB_DIR=%ROOT_DIR%apps\web"

echo ==============================================================================
echo        EMOTION DETECTION AI - PRODUCTION ENVIRONMENT (PROD BUILD)
echo ==============================================================================
echo.

:: 1. Check Python Virtual Environment
if not exist "%PYTHON_EXE%" (
    echo [!] Virtual environment python not found at: %PYTHON_EXE%
    echo [*] Checking system PATH for python...
    where python >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_EXE=python"
        echo [OK] Using system python.
    ) else (
        echo [ERROR] Python not found! Please create .venv or install Python.
        pause
        exit /b 1
    )
) else (
    echo [OK] Python detected: %PYTHON_EXE%
)

:: 2. Check Node / Package Manager
where pnpm >nul 2>nul
if %errorlevel% equ 0 (
    set "PKG_MGR=pnpm"
    echo [OK] Package manager detected: pnpm
) else (
    where npm >nul 2>nul
    if %errorlevel% equ 0 (
        set "PKG_MGR=npm run"
        echo [OK] Package manager detected: npm
    ) else (
        echo [ERROR] Neither pnpm nor npm was found in PATH!
        pause
        exit /b 1
    )
)

:: 3. Check / Run Next.js Production Build
echo.
if not exist "%WEB_DIR%\.next" (
    echo [*] No existing production build found in apps/web/.next.
    echo [*] Building optimized Next.js production bundle now...
    cd /d "%WEB_DIR%"
    call %PKG_MGR% build
    if errorlevel 1 (
        echo [ERROR] Next.js production build failed!
        pause
        exit /b 1
    )
    cd /d "%ROOT_DIR%"
) else (
    echo [OK] Production build already available in apps/web/.next.
)

echo.
echo ------------------------------------------------------------------------------
echo Starting Production Services...
echo  - Backend API:  FastAPI with Auto-Reload on Code Changes (Port 8000)
echo  - Frontend Web: Next.js Optimized Production Server (Port 3000)
echo ------------------------------------------------------------------------------
echo.

:: 4. Launch Backend in a dedicated titled window (with auto-reload on code changes)
start "Emotion AI - Backend (FastAPI Prod)" cmd /k "cd /d "%API_DIR%" && echo [*] Starting FastAPI with Auto-Reload on http://0.0.0.0:8000 ... && "%PYTHON_EXE%" -m uvicorn app.main:app --reload --reload-dir . --reload-dir ../../ml --host 0.0.0.0 --port 8000"

:: Wait 2 seconds before launching frontend
timeout /t 2 /nobreak >nul

:: 5. Launch Frontend in a dedicated titled window (production server)
start "Emotion AI - Frontend (Next.js Prod)" cmd /k "cd /d "%WEB_DIR%" && echo [*] Starting Next.js Production Server on http://localhost:3000 ... && %PKG_MGR% start"

echo.
echo ==============================================================================
echo                 PRODUCTION SERVICES SUCCESSFULLY LAUNCHED!
echo ==============================================================================
echo.
echo   Frontend (UI):       http://localhost:3000
echo   Backend (API):      http://localhost:8000
echo   Interactive Docs:   http://localhost:8000/docs
echo   Alternative Docs:   http://localhost:8000/redoc
echo   Health Check:       http://localhost:8000/api/v1/health
echo.
echo   Auto-Reload:        ENABLED (Backend reloads on changes in apps/api and ml/)
echo.
echo ==============================================================================
echo Press [Enter] or any key to open the Web App in your browser (http://localhost:3000)...
pause >nul
start http://localhost:3000
exit /b 0
