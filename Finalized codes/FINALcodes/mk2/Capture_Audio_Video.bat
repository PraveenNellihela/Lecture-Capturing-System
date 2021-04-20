@echo off
set arg1=%1
call activate tensorflow1
start idle -r videorecorder_FINAL.py %arg1%
start idle -r audiorecorder_FINAL.py %arg1%



