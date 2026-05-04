@echo off
cd /d "D:\claude code\ashare_screener"
echo [%date% %time%] 开始每日更新... >> daily_run.log
"D:\anaconda\python.exe" run.py ingest >> daily_run.log 2>&1
"D:\anaconda\python.exe" run.py pick >> daily_run.log 2>&1
echo [%date% %time%] 完成 >> daily_run.log
