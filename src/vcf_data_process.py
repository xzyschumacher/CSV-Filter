import pandas as pd
import re

data_dir = "../data/"


def list_save(filename, data):#filename为写入文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    file.writelines(data)
    file.close()
    print(filename + "文件保存成功")

def set_save(filename, data):#filename为写入文件的路径，data为要写入数据列表.
    file = open(filename,'w')
    file.writelines([line+'\n' for line in data])
    file.close()
    print(filename + "文件保存成功")

insert = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA12878\n"]
delete = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA12878\n"]

filename = data_dir + "NA12878.sorted.vcf"

chr_list = set()

with open(filename, "r") as f:
    lines = f.readlines()
    for data in lines:
        if "#" in data:
            if "contig=<ID=" in data:
                chr_list.add(re.split("=|,", data)[2])
        else:
            if "DEL" in data:
                delete.append(data)
            elif "INS" in data:
                insert.append(data)

list_save(filename + "_ins", insert)
list_save(filename + "_del", delete)
set_save(filename + "_chr", chr_list)

insert_result_data = pd.read_csv(filename + "_ins", sep = "\t")

insert_result_data.insert(2,'SPOS',0)
insert_result_data.insert(3,'EPOS',0)
insert_result_data.insert(4,'SVLEN',0)

for index, row in insert_result_data.iterrows():
    print(index)
    #SPOS, EPOS
    s = row["INFO"]
    pos = s.find("CIPOS")
    if pos != -1:
        pos = pos + 6 # "CIPOS="
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        insert_result_data.loc[index, ["SPOS"]] = start
        insert_result_data.loc[index, ["EPOS"]] = end

    # END
    s = row["INFO"]
    pos = s.find("SVLEN")
    if pos == -1:
        pos = s.find("END") + 4 # "END="
        s = s[pos:]
        s = s.split(";")[0]
        s = int(s) - row["POS"]
        insert_result_data.loc[index, ["SVLEN"]] = s
    else:
        pos = pos + 6 # "SVLEN="
        s = s[pos:]
        s = s.split(";")[0]
        s = int(s)
        insert_result_data.loc[index, ["SVLEN"]] = s


insert_result_data.to_csv(data_dir + "insert_result_data.csv.vcf", sep="\t")



delete_result_data = pd.read_csv(filename + "_del", sep = "\t")

delete_result_data.insert(2,'SPOS',0)
delete_result_data.insert(3,'EPOS',0)
delete_result_data.insert(4,'END',0)
delete_result_data.insert(5,'SEND',0)
delete_result_data.insert(6,'EEND',0)

for index, row in delete_result_data.iterrows():
    print(index)
    #SPOS, EPOS
    s = row["INFO"]
    pos = s.find("CIPOS")
    if pos != -1:
        pos = pos + 6 # "CIPOS="
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        delete_result_data.loc[index, ["SPOS"]] = start
        delete_result_data.loc[index, ["EPOS"]] = end

    # END
    s = row["INFO"]
    pos = s.find("END") + 4 # "END="
    s = s[pos:]
    s = s.split(";")[0]
    s = int(s)
    delete_result_data.loc[index, ["END"]] = s

    #SPOS, EPOS
    s = row["INFO"]
    pos = s.find("CIEND")
    if pos != -1:
        pos = pos + 6 # "CIEND="
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        delete_result_data.loc[index, ["SEND"]] = start
        delete_result_data.loc[index, ["EEND"]] = end

delete_result_data.to_csv(data_dir + "delete_result_data.csv.vcf", sep="\t")


# index creation
# parallel  samtools index ::: *.bam
