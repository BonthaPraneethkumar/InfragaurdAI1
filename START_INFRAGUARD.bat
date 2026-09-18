@echo off
setlocal
cd /d "%~dp0"
title InfraGuard AI
echo ==========================================
echo          InfraGuard AI - START
echo ==========================================
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py")
if not defined PY (
  for /d %%D in ("%LOCALAPPDATA%\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_*") do (
    if exist "%%D\python.exe" set "PY=%%D\python.exe"
  )
)
if not defined PY (
  echo Python was not found.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python environment...
  "%PY%" -m venv .venv
  if errorlevel 1 goto fail
)
call ".venv\Scripts\activate.bat"
echo Installing project packages...
python -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 goto fail
if not exist ".env" copy ".env.example" ".env" >nul
echo Starting InfraGuard AI...
start "" "http://127.0.0.1:8000"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
goto end
:fail
echo Something failed. Take a screenshot and send it to ChatGPT.
pause
:end
endlocal
