@echo off
echo Starting AI Assistant Application...
python start_application.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo Error occurred while running the application.
    echo Please check the error message above.
    pause
    exit /b %ERRORLEVEL%
)
