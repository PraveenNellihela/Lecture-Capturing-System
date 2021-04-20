import subprocess
import random

filepath = r 'C:\tensorflow1\models\research\object_detection\scheduledAutoRun.bat'
item = subprocess.Popen([filepath, 'arg1'], shell=True)
for line in item.stdout:
     print line
