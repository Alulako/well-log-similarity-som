@echo off
cd /d "%~dp0"
python validate_experiment.py
if errorlevel 1 exit /b 1
python experiment.py --save
if errorlevel 1 exit /b 1
echo.
echo Experimento concluido. Abra results\report.html para ver o resumo.
pause
