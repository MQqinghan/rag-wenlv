@echo off
cd /d %~dp0
.venv\Scripts\python.exe app\api\http\query_server.py
