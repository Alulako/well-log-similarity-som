@echo off
cd /d "%~dp0"
python experiment.py --save
echo.
echo Experimento concluido. Abra results\report.html para ver o resumo.
pause
