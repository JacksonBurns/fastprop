
# performs 4 repeats of chemprop (with the same seeds as fastprop, FWIW)

chemprop_train \
--data_path benchmark_data.csv \
--target_columns fraction \
--smiles_column SMILES \
--dataset_type regression \
--save_dir chemprop_fubrain_12345_logs \
--epochs 400 \
--split_sizes 0.64 0.07 0.29 \
--metric rmse \
--extra_metrics mae \
--batch_size 256 \
--seed 12345

chemprop_train \
--data_path benchmark_data.csv \
--target_columns fraction \
--smiles_column SMILES \
--dataset_type regression \
--save_dir chemprop_fubrain_12346_logs \
--epochs 400 \
--split_sizes 0.64 0.07 0.29 \
--metric rmse \
--extra_metrics mae \
--batch_size 256 \
--seed 12346

chemprop_train \
--data_path benchmark_data.csv \
--target_columns fraction \
--smiles_column SMILES \
--dataset_type regression \
--save_dir chemprop_fubrain_12347_logs \
--epochs 400 \
--split_sizes 0.64 0.07 0.29 \
--metric rmse \
--extra_metrics mae \
--batch_size 256 \
--seed 12347

chemprop_train \
--data_path benchmark_data.csv \
--target_columns fraction \
--smiles_column SMILES \
--dataset_type regression \
--save_dir chemprop_fubrain_12348_logs \
--epochs 400 \
--split_sizes 0.64 0.07 0.29 \
--metric rmse \
--extra_metrics mae \
--batch_size 256 \
--seed 12348