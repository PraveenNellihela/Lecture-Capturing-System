from subprocess import Popen
p = Popen("scheduledAutoRun.bat", cwd=r"C:\tensorflow1\models\research\object_detection\scheduledAutoRun")
stdout, stderr = p.communicate()
