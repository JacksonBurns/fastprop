import sys

source = "HIV.csv"

ternary_encoding = dict(CA="0", CM="1", CI="2")
with open(source, "r") as file:
    data = file.readlines()
if len(sys.argv) == 1:
    print("smiles,ternary_activity,binary_activity")
data.pop(0)
for line in data:
    line = line.replace("\n", "")
    if not line:
        continue
    smi, tern, bin = line.split(",")
    if len(sys.argv) == 1:
        print(smi, ternary_encoding[tern], bin, sep=",")
    else:
        print(smi)
