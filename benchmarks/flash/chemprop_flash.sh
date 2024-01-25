# usage note: fastprop will automatically filter out dataset entries with missing target
# values, but chemprop requires this to be done beforehand.
# You can do this with grep: grep -v NA benchmark_data.csv > chemprop_data.csv
# or with your spreadsheet software of choice.

chemprop_train \
--data_path chemprop_data.csv \
--smiles_columns smiles \
--target_columns flash \
--dataset_type regression \
--save_dir chemprop_flash_195091191_logs \
--epochs 300 \
--split_sizes 0.7 0.2 0.1 \
--metric mae \
--extra_metrics rmse \
--batch_size 1024 \
--seed 195091191

chemprop_train \
--data_path chemprop_data.csv \
--smiles_columns smiles \
--target_columns flash \
--dataset_type regression \
--save_dir chemprop_flash_195091192_logs \
--epochs 300 \
--split_sizes 0.7 0.2 0.1 \
--metric mae \
--extra_metrics rmse \
--batch_size 1024 \
--seed 195091192

chemprop_train \
--data_path chemprop_data.csv \
--smiles_columns smiles \
--target_columns flash \
--dataset_type regression \
--save_dir chemprop_flash_195091193_logs \
--epochs 300 \
--split_sizes 0.7 0.2 0.1 \
--metric mae \
--extra_metrics rmse \
--batch_size 1024 \
--seed 195091193

chemprop_train \
--data_path chemprop_data.csv \
--smiles_columns smiles \
--target_columns flash \
--dataset_type regression \
--save_dir chemprop_flash_195091194_logs \
--epochs 300 \
--split_sizes 0.7 0.2 0.1 \
--metric mae \
--extra_metrics rmse \
--batch_size 1024 \
--seed 195091194