@echo off
setlocal
cd /d "%~dp0"
title StockfishNN - Game Replay
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" replay.py
if errorlevel 1 (
  echo.
  echo Replay viewer exited with an error.
  pause
)
endlocal
