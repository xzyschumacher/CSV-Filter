from datetime import datetime
from threading import Timer
# 打印时间函数
import subprocess

cmd = "kill -9 "

def printTime(inc):
    d = subprocess.getoutput("ps -ef | grep train").split('\n')
    for id in d:
        id = id.split()
        if id[2] == '1341':
            print(cmd + id[1])
            subprocess.call(cmd + id[1], shell=True)
    # print(command1)
    # subprocess.call(command1, shell=True)
    # command2 = 'git commit -m "commit ' + str(count) + '" -a'
    # print(command2)
    # subprocess.call(command2, shell=True)
    # print(command3)
    # subprocess.call(command3, shell=True)
    t = Timer(inc, printTime, (inc,))
    t.start()


# 1m
printTime(60)
