from datetime import datetime
from threading import Timer
import subprocess

cmd = "kill -9 "

def printTime(inc):
    d = subprocess.getoutput("ps -ef | grep train").split('\n')
    for id in d:
        id = id.split()
        if id[2] == '1808':
            print(cmd + id[1])
            subprocess.call(cmd + id[1], shell=True)
    t = Timer(inc, printTime, (inc,))
    t.start()


# 1m
printTime(60)
