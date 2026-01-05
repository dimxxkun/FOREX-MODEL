@echo off
echo ============================================================
echo FOREX SIGNAL GENERATOR
echo ============================================================
echo.

cd /d "%~dp0"
python signal_generator.py --account 10000

echo.
echo ============================================================
pause
