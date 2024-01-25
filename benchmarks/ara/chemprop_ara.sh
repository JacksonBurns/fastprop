
# performs 4 repeats of chemprop (with the same seeds as fastprop, FWIW)

chemprop_train \
--data_path benchmark_data.csv \
--target_columns Activity \
--smiles_column Smiles \
--dataset_type classification \
--save_dir chemprop_ara_1989_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--extra_metrics accuracy \
--batch_size 1024 \
--seed 1989

chemprop_train \
--data_path benchmark_data.csv \
--target_columns Activity \
--smiles_column Smiles \
--dataset_type classification \
--save_dir chemprop_ara_1990_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--extra_metrics accuracy \
--batch_size 1024 \
--seed 1990

chemprop_train \
--data_path benchmark_data.csv \
--target_columns Activity \
--smiles_column Smiles \
--dataset_type classification \
--save_dir chemprop_ara_1991_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--extra_metrics accuracy \
--batch_size 1024 \
--seed 1991

chemprop_train \
--data_path benchmark_data.csv \
--target_columns Activity \
--smiles_column Smiles \
--dataset_type classification \
--save_dir chemprop_ara_1992_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--extra_metrics accuracy \
--batch_size 1024 \
--seed 1992