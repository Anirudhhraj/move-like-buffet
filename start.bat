@echo off
echo ============================================
echo   Buffett Bureaucracy - Starting Services
echo ============================================
echo.

echo [1/2] Starting backend (port 8000)...
start "Backend" cmd /k "cd /d %~dp0backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo [2/2] Starting frontend (port 5173)...
timeout /t 3 /nobreak >nul
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both services starting in separate windows.
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo.
echo Close this window anytime - services keep running.
pause