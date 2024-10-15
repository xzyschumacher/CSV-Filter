import pandas as pd
import re

data_dir = "../data/"

def list_save(filename, data):
    file = open(filename,'w')
    file.writelines(data)
    file.close()
    print(filename + " file saved successfully")

def set_save(filename, data):
    file = open(filename,'w')
    file.writelines([line+'\n' for line in data])
    file.close()
    print(filename + " file saved successfully")

insert = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG002\n"]
delete = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG002\n"]

filename = data_dir + "sniffles2-CLR-minimap2.vcf"

chr_list = set()

with open(filename, "r") as f:
    lines = f.readlines()
    for data in lines:
        if "#" in data:
            if "contig=<ID=" in data:
                chr_list.add(re.split("=|,", data)[2])
        else:
            if "SVTYPE=DEL" in data:
                delete.append(data)
            elif "SVTYPE=INS" in data:
                insert.append(data)

list_save(filename + "_ins", insert)
list_save(filename + "_del", delete)
set_save(filename + "_chr", chr_list)

insert_result_data = pd.read_csv(filename + "_ins", sep = "\t")

insert_result_data.insert(2,'SPOS',0)
insert_result_data.insert(3,'EPOS',0)
insert_result_data.insert(4,'SVLEN',0)

for index, row in insert_result_data.iterrows():
    print(f"INS index = {index}", end='\r')
    s = row["INFO"]
    pos = s.find("CIPOS")
    if pos != -1:
        pos = pos + 6
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        insert_result_data.loc[index, ["SPOS"]] = start
        insert_result_data.loc[index, ["EPOS"]] = end
    else:
        insert_result_data.loc[index, ["SPOS"]] = 0
        insert_result_data.loc[index, ["EPOS"]] = 0

    s = row["INFO"]
    pos = s.find("SVLEN")
    if pos == -1:
        pos = s.find("END") + 4
        s = s[pos:]
        s = s.split(";")[0]
        s = int(s) - row["POS"]
        insert_result_data.loc[index, ["SVLEN"]] = s
    else:
        pos = pos + 6
        s = s[pos:]
        s = s.split(";")[0]
        s = int(s)
        insert_result_data.loc[index, ["SVLEN"]] = s

insert_result_data.to_csv(data_dir + "insert_result_data.csv.vcf", sep="\t")

print(f"INS finished, total number = {index}")

delete_result_data = pd.read_csv(filename + "_del", sep = "\t")

delete_result_data.insert(2,'SPOS',0)
delete_result_data.insert(3,'EPOS',0)
delete_result_data.insert(4,'END',0)
delete_result_data.insert(5,'SEND',0)
delete_result_data.insert(6,'EEND',0)

for index, row in delete_result_data.iterrows():
    print(f"DEL index = {index}", end='\r')
    s = row["INFO"]
    pos = s.find("CIPOS")
    if pos != -1:
        pos = pos + 6 
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        delete_result_data.loc[index, ["SPOS"]] = start
        delete_result_data.loc[index, ["EPOS"]] = end
    else:
        insert_result_data.loc[index, ["SPOS"]] = 0
        insert_result_data.loc[index, ["EPOS"]] = 0

    s = row["INFO"]
    pos = s.find("END")
    if pos != -1:
        pos += 4
        s_end = s[pos:]
        s_end = s_end.split(";")[0]
        try:
            end_value = int(s_end)
        except ValueError:
            print(f"Error parsing END value: {s_end}")
            end_value = None
    else:
        pos = s.find("SVLEN")
        if pos != -1:
            pos +=6
            s_svlen = s[pos:]
            s_svlen = s_svlen.split(";")[0]
            try:
                svlen_value = abs(int(s_svlen))
                end_value = row["POS"] + svlen_value
                value_tmp = row["POS"]
            except ValueError:
                print(f"Error parsing SVLEN value: {s_svlen}")
                end_value = None
        else:
            print(f"Neither END nor SVLEN found in INFO: {s}")
            end_value = None

    if end_value is not None:
        delete_result_data.loc[index, ["END"]] = end_value
    else:
        print(f"Unable to set END for index {index}")

    s = row["INFO"]
    pos = s.find("CIEND")
    if pos != -1:
        pos = pos + 6 
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        delete_result_data.loc[index, ["SEND"]] = start
        delete_result_data.loc[index, ["EEND"]] = end
    else:
        delete_result_data.loc[index, ["SEND"]] = 0
        delete_result_data.loc[index, ["EEND"]] = 0

delete_result_data.to_csv(data_dir + "delete_result_data.csv.vcf", sep="\t")

print(f"DEL finished, total number = {index}")
