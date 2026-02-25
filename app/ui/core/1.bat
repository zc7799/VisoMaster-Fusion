@echo off
setlocal enabledelayedexpansion

:: 定义绝对路径（根据实际项目结构调整）
set "BASE_DIR=E:\VisoMaster-Fusion"
set "UI_FILE=%BASE_DIR%\app\ui\core\MainWindow.ui"
set "PY_FILE=%BASE_DIR%\app\ui\core\main_window.py"
set "QRC_FILE=%BASE_DIR%\app\ui\core\media.qrc"
set "RCC_PY_FILE=%BASE_DIR%\app\ui\core\media_rc.py"

:: 激活conda环境（确保在此处激活）
call conda activate visomaster

:: 执行转换命令（使用绝对路径）
"%CONDA_PREFIX%\Scripts\pyside6-uic.exe" "%UI_FILE%" -o "%PY_FILE%"
"%CONDA_PREFIX%\Scripts\pyside6-rcc.exe" "%QRC_FILE%" -o "%RCC_PY_FILE%"

:: 替换导入语句（保持原有逻辑不变）
set "searchString=import media_rc"
set "replaceString=from app.ui.core import media_rc"
set "tempFile=%PY_FILE%.tmp"

(for /f "usebackq delims=" %%A in ("%PY_FILE%") do (
    set "line=%%A"
    if "!line!"=="%searchString%" (
        echo %replaceString%
    ) else (
        echo !line!
    )
)) > "%tempFile%"
move /y "%tempFile%" "%PY_FILE%"

echo Conversion completed successfully.
pause