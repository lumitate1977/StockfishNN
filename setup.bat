@echo off
setlocal
cd /d "%~dp0"
title StockfishNN - Setup
echo Setting up StockfishNN...
if not exist ".venv\Scripts\python.exe" (
  py -3.13 -m venv .venv 2>nul
  if errorlevel 1 py -3 -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Could not create the virtual environment.
  echo Install Python 3.11 or newer, then run setup.bat again.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo Setup complete.
echo 1. Set stockfish_path in config.json to your stockfish.exe location.
echo 2. Run train.bat to train, or train_forever.bat for continuous training.
echo 3. Run play.bat to play or replay.bat to review saved games.
pause
exit /b 0
:failed
echo.
echo Dependency installation failed. Check the error above.
pause
exit /b 1
