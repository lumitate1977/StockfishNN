@echo off
setlocal
cd /d "%~dp0"
title StockfishNN - Efficient Training
rem Keep CPU use balanced between Stockfish and PyTorch.
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=1
if not exist ".venv\Scripts\python.exe" (
  echo Python environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
echo StockfishNN efficient training
 echo - Resumes models\latest.pt
 echo - Uses separate move-quality and whole-game rewards
 echo - 8 games/cycle, 2 epochs, batch size 128
 echo - Stockfish skill 5 / depth 5, 2 threads, 512 MB hash
 echo - PyTorch uses 4 CPU threads; replay arrays are packed in RAM for faster training
 echo - Stores new labels in data\replay_buffer_move_game_v3.npz (old replay buffer is preserved)
echo - Runs a 6-game benchmark every 10 cycles
 echo.
".venv\Scripts\python.exe" train.py --cycles 10 --games-per-cycle 8 --epochs 2 --batch-size 128 --max-plies 120 --max-samples 2500 --skill 5 --depth 5 --threads 2 --hash-mb 512 --torch-threads 4 --benchmark-every 10 --benchmark-games 6
if errorlevel 1 (
  echo.
  echo Training exited with an error. Check the message above.
  pause
) else (
  echo.
  echo Training run complete. Latest checkpoint was saved.
  pause
)
endlocal
