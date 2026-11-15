@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main_menu.py
) else (
    python main_menu.py
)
if errorlevel 1 pause
