@echo off
echo === Python asennus ===

REM Tarkista Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Python ei loytynyt! Asenna Python ensin:
    echo https://www.python.org/downloads/
    pause
    exit /b
)

echo Asennetaan kirjastot...
pip install --upgrade pip
pip install -r requirements.txt

echo --- Valmis! ---
pause

