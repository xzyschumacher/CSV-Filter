import os
from datetime import datetime
import subprocess

command1 = "git add ."
command2 = 'git commit -m "commit" -a'
command3 = "git push origin main"

count = 0
commit_count = 0


for root, dirs, files in os.walk(".", topdown=False):
    for name in files:

        print()
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        path = os.path.join(root, name)
        print(path)
        if path.find("/.") != -1:
            continue
        if os.path.getsize(path)/1024/1024 > 100:
            continue

        command1 = "git add " + path

        print()
        print(command1)
        subprocess.call(command1, shell=True)
        commit_count += 1

        if commit_count == 16:
            commit_count = 0
            command2 = 'git commit -m "commit init ' + str(count) + '"'
            count += 1
            error_flag = "nothing added to commit but untracked files present"

            print()
            print(command2)
            command2_exitcode, command2_output = subprocess.getstatusoutput(
                command2)
            if command2_output.find(error_flag) != -1:
                print(command2_output.split("\n")[-1])
                continue
            else:
                print(command2_output)

            print()
            print(command3)
            subprocess.call(command3, shell=True)
            