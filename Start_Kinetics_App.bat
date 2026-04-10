@echo off
setlocal

cd /d "%~dp0"
title Kinetics_app Launcher

set "PYTHON_EXE="
set "PYTHON_ARGS="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 --version >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=py"
            set "PYTHON_ARGS=-3"
        )
    )
)

if not defined PYTHON_EXE (
    for /f "delims=" %%I in ('where.exe python 2^>nul') do (
        call :try_python_path "%%~fI"
        if not errorlevel 1 goto :python_found
    )
)

:python_found
if not defined PYTHON_EXE (
    echo [ERROR] Python 3.10+ was not found.
    echo Install Python and enable "Add python.exe to PATH".
    echo Then double-click this file again.
    pause
    exit /b 1
)

call :run_python --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was detected but could not be started.
    echo Install a working Python 3.10+ runtime, then try again.
    pause
    exit /b 1
)

if defined PYTHON_ARGS (
    echo [INFO] Using Python launcher: %PYTHON_EXE% %PYTHON_ARGS%
) else (
    echo [INFO] Using Python: %PYTHON_EXE%
)

echo [INFO] Checking Python version...
call :run_python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo [ERROR] Python 3.10+ is required to start Kinetics_app.
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...
call :run_python -c "import streamlit, numpy, pandas, scipy, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing requirements. First launch may take a few minutes...
    call :run_python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements.
        echo Check your network connection and Python/pip setup, then try again.
        pause
        exit /b 1
    )
)

echo [INFO] Starting app...
echo [INFO] If the browser does not open automatically, go to http://localhost:8501
echo.
call :run_python -m streamlit run main.py --server.headless false
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] App exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%

:try_python_path
echo %~1 | find /I "\WindowsApps\" >nul
if not errorlevel 1 exit /b 1

"%~1" --version >nul 2>nul
if errorlevel 1 exit /b 1

set "PYTHON_EXE=%~1"
set "PYTHON_ARGS="
exit /b 0

:run_python
if defined PYTHON_ARGS (
    %PYTHON_EXE% %PYTHON_ARGS% %*
) else (
    "%PYTHON_EXE%" %*
)
exit /b %ERRORLEVEL%
