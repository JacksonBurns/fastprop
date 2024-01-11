source = "database.txt"

with open(source, "r") as file:
    data = file.readlines()
# throw out the comment lines
data = data[3:]
print("SMILES,exp,exp_unc,calc,calc_unc")
for line in data:
    _, smiles, _, exp, exp_unc, calc, calc_unc, *_ = line.split("; ")
    print(smiles, exp, exp_unc, calc, calc_unc, sep=",")
