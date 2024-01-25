source = "HOPV_15_revised_2.data"

with open(source, "r") as file:
    data = file.readlines()
print("smiles,doi,inchi_key,construction,architecture,complement,homo,lumo,echem_gap,opt_gap,pce,voc,jsc,fill_factor")
counter = 0
res = ""
for line in data:
    if res:
        print(line[:-1], res[:-1], sep=",")
        res = ""
    if line.startswith("10."):
        # skip those with missing PCE
        if line.split(",")[-3] == "nan":
            continue
        res = line
