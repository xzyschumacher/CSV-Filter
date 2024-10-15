import subprocess
import os
from utilities import mymkdir

bam_name = "HG002-ONT-minimap2.sorted.bam"
output_file = "output.depth.txt"
bam_data_dir = "../data/"
data_dir = "../data/"

cmd = "samtools depth " + bam_data_dir + bam_name + " > " + data_dir + output_file
print(cmd)
print("==== starting samtools deal ====")
subprocess.call(cmd, shell = True)

mymkdir(data_dir + "depth/")

with open(data_dir + output_file, "r") as f:
    for line in f:
        with open(data_dir + "depth/" + line.split("\t")[0], "a+") as subf:
            subf.write(line)
