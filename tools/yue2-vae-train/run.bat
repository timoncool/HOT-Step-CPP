@echo off
rem YuE2 2x decoder fine-tune. Pass --resume to continue the run after a STOP.
rem   Stop cleanly:  create a file named STOP in %RUN%  (saves, then exits)
rem   Progress:      %RUN%\train.log, %RUN%\evals\report.md, %RUN%\evals\audio\
setlocal
set ROOT=D:\Ace-Step-Latest\vae-training
set Y=%ROOT%\yue2
set RUN=%Y%\runs\dec2x
if exist "%RUN%\STOP" del "%RUN%\STOP"
cd /d "%~dp0"
"%ROOT%\venv\Scripts\python.exe" train.py --base "%Y%\base" --run "%RUN%" --evalset "%Y%\split\evalset.npz" --train-list "%Y%\split\train.txt" --n-res 9 --batch 6 --steps 60000 --eval-every 2000 --vram-gb 26 %*
