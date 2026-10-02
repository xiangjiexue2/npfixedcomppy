@echo off
cd /d C:\Users\xxjie\Documents\rebuild\npfixedcomppy
call .venv\Scripts\python.exe setup.py build_ext 2>&1
echo build_exit=%errorlevel%
