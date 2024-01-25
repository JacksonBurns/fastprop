chemprop_train \
--data_path benchmark_data.csv \
--smiles_columns anion_smiles cation_smiles \
--number_of_molecules 2 \
--features_path chemprop_features.csv \
--target_columns solubility \
--dataset_type regression \
--save_dir chemprop_h2s_62_logs \
--epochs 200 \
--split_sizes 0.8 0.1 0.1 \
--metric rmse \
--batch_size 1024 \
--seed 62

chemprop_train \
--data_path benchmark_data.csv \
--smiles_columns anion_smiles cation_smiles \
--number_of_molecules 2 \
--features_path chemprop_features.csv \
--target_columns solubility \
--dataset_type regression \
--save_dir chemprop_h2s_63_logs \
--epochs 200 \
--split_sizes 0.8 0.1 0.1 \
--metric rmse \
--batch_size 1024 \
--seed 63
