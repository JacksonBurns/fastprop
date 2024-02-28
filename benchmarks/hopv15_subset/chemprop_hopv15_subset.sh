COUNTER=1

while [ $COUNTER -le 15 ];
do
    chemprop_train \
    --data_path benchmark_data.csv \
    --smiles_columns smiles \
    --target_columns pce \
    --dataset_type regression \
    --save_dir chemprop_hopv15_subset_${COUNTER}_logs \
    --epochs 100 \
    --split_sizes 0.6 0.2 0.2 \
    --metric mae \
    --extra_metrics rmse \
    --batch_size 1024 \
    --seed $COUNTER
    COUNTER=$((COUNTER+1))
done
