@echo off
cd /d "D:\ClaudeCODE"
echo [%date% %time%] 开始每日更新... >> daily_run.log
"C:\Users\miih\AppData\Local\Programs\Python\Python39\python.exe" run.py ingest >> daily_run.log 2>&1
"C:\Users\miih\AppData\Local\Programs\Python\Python39\python.exe" run.py pick >> daily_run.log 2>&1
echo [%date% %time%] 完成 >> daily_run.log
