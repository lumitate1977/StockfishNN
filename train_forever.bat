@echo off
setlocal
cd /d "%~dp0"
title StockfishNN - Efficient Continuous Training
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=1
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
echo Efficient continuous training with move + game rewards is starting.
echo It resumes models\latest.pt and saves after every cycle.
echo Rewards: 75%% individual move quality + 25%% whole-game outcome/evaluation.
echo CPU profile: Stockfish 2 threads, PyTorch 4 threads, 512 MB hash.
echo Press Ctrl+C to stop; the last completed cycle remains saved.
echo.
".venv\Scripts\python.exe" train.py --cycles 1000000 --games-per-cycle 8 --epochs 2 --batch-size 128 --max-plies 120 --max-samples 2500 --skill 5 --depth 5 --threads 2 --hash-mb 512 --torch-threads 4 --benchmark-every 10 --benchmark-games 6
if errorlevel 1 (
  echo.
  echo Training stopped or exited with an error. Check the message above.
  pause
)
endlocal
