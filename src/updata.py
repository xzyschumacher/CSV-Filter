import os
from datetime import datetime
# 打印时间函数
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


# git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch .cache/pip/http/0/8/f/7/7/08f7798e86a86d6ff55c79fefd6a3e1980488663023c57c42c4d8189' --prune-empty --tag-name-filter cat -- --all

# remote: error: File .cache/huggingface/transformers/58592490276d9ed1e8e33f3c12caf23000c22973cb2b3218c641bd74547a1889.fabda197bfe5d6a318c2833172d6757ccc7e49f692cb949a6fabf560cee81508 is 392.51 MB; this exceeds GitHub's file size limit of 100.00 MB
