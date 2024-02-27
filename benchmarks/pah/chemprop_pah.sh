COUNTER=55

while [ $COUNTER -le 62 ];
do
    chemprop_train \
    --data_path arockiaraj_pah_data.csv \
    --smiles_columns smiles \
    --target_columns log_p \
    --dataset_type regression \
    --save_dir chemprop_pah_${COUNTER}_logs \
    --epochs 40 \
    --split_sizes 0.8 0.1 0.1 \
    --metric mae \
    --extra_metrics rmse r2 \
    --batch_size 64 \
    --seed $COUNTER
    COUNTER=$((COUNTER+1))
done
