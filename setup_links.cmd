@echo off
REM ==========================================================
REM  重建共享层目录链接（junction），让快照结构与本地开发一致
REM  用法：把本文件与 retrieval/ import/ shared/ 放在同一目录，双击或命令行执行
REM  安全约定：只创建缺失的链接；已存在的目录一律跳过，绝不删除
REM ==========================================================
setlocal
set "ROOT=%~dp0"
set "SHARED=%ROOT%shared\app"

call :link "%ROOT%retrieval\app\shared"      "%SHARED%\shared"
call :link "%ROOT%retrieval\app\infra"       "%SHARED%\infra"
call :link "%ROOT%retrieval\app\rag\common" "%SHARED%\rag\common"
call :link "%ROOT%import\app\shared"         "%SHARED%\shared"
call :link "%ROOT%import\app\infra"          "%SHARED%\infra"
call :link "%ROOT%import\app\rag\common"    "%SHARED%\rag\common"

echo.
echo 完成。若两个域需要共用同一个虚拟环境，可另行执行（目标路径按需替换）：
echo   mklink /J "%ROOT%retrieval\.venv" "%%USERPROFILE%%\...\venv"
echo.
goto :eof

:link
if not exist "%~2" (
  echo [WARN] 共享层源目录不存在，跳过：%~2
  exit /b 0
)
if exist "%~1" (
  echo [SKIP] 已存在，未改动：%~1
  exit /b 0
)
mklink /J "%~1" "%~2"
exit /b 0
