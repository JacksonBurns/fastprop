# performs 5 repeats of chemprop (with the same seeds as fastprop, FWIW)

chemprop_train \
--data_path benchmark_data.csv \
--target_columns se1 se2 se3 se4 se5 se6 se7 se8 se9 se10 se11 se12 se13 se14 se15 se16 se17 se18 se19 se20 se21 se22 se23 se24 se25 se26 se27 \
--smiles_column smiles \
--dataset_type classification \
--save_dir chemprop_sider_19716_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--batch_size 2048 \
--seed 19716

chemprop_train \
--data_path benchmark_data.csv \
--target_columns se1 se2 se3 se4 se5 se6 se7 se8 se9 se10 se11 se12 se13 se14 se15 se16 se17 se18 se19 se20 se21 se22 se23 se24 se25 se26 se27 \
--smiles_column smiles \
--dataset_type classification \
--save_dir chemprop_sider_19717_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--batch_size 2048 \
--seed 19717

chemprop_train \
--data_path benchmark_data.csv \
--target_columns se1 se2 se3 se4 se5 se6 se7 se8 se9 se10 se11 se12 se13 se14 se15 se16 se17 se18 se19 se20 se21 se22 se23 se24 se25 se26 se27 \
--smiles_column smiles \
--dataset_type classification \
--save_dir chemprop_sider_19718_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--batch_size 2048 \
--seed 19718

chemprop_train \
--data_path benchmark_data.csv \
--target_columns se1 se2 se3 se4 se5 se6 se7 se8 se9 se10 se11 se12 se13 se14 se15 se16 se17 se18 se19 se20 se21 se22 se23 se24 se25 se26 se27 \
--smiles_column smiles \
--dataset_type classification \
--save_dir chemprop_sider_19719_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--batch_size 2048 \
--seed 19719

chemprop_train \
--data_path benchmark_data.csv \
--target_columns se1 se2 se3 se4 se5 se6 se7 se8 se9 se10 se11 se12 se13 se14 se15 se16 se17 se18 se19 se20 se21 se22 se23 se24 se25 se26 se27 \
--smiles_column smiles \
--dataset_type classification \
--save_dir chemprop_sider_19720_logs \
--loss_function binary_cross_entropy \
--epochs 300 \
--split_sizes 0.8 0.1 0.1 \
--metric auc \
--batch_size 2048 \
--seed 19720