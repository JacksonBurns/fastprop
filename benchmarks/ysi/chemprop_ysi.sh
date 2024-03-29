COUNTER=1

while [ $COUNTER -le 8 ];
do
    chemprop_train \
    --data_path benchmark_data.csv \
    --smiles_columns SMILES \
    --target_columns YSI \
    --dataset_type regression \
    --save_dir chemprop_ysi_${COUNTER}_logs \
    --epochs 300 \
    --split_sizes 0.6 0.2 0.2 \
    --metric mae \
    --extra_metrics rmse \
    --batch_size 1024 \
    --seed $COUNTER
    COUNTER=$((COUNTER+1))
done
