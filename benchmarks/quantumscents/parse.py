from dataset.loader import CID_TO_SMILES, QS_DATA

QS_DATA.insert(0, "smiles", list(CID_TO_SMILES[i] for i in QS_DATA.index))

QS_DATA.rename(columns={"black currant": "black_currant"}, inplace=True)

QS_DATA.to_csv("benchmark_data.csv")
