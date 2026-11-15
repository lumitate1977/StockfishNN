@echo off
setlocal
cd /d "%~dp0"
title StockfishNN - Play
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
if not exist "models\latest.pt" (
  echo No trained model was found at models\latest.pt.
  echo Run train.bat first, then try again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" play_gui.py
if errorlevel 1 (
  echo.
  echo Play window exited with an error.
  pause
)
endlocal
