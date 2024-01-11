from py2opsin import py2opsin

source = "Alkane/dataset_boiling_point_names.txt"

with open(source, "r") as file:
    data = file.readlines()
temp = [line.split() for line in data]
smiles = py2opsin([i[3].replace("\n", "") for i in temp])
print("index,boiling_point,py2opsin_smiles")
for data, smile in zip(temp, smiles):
    print(f"{data[0]},{data[2]},{smile}")
