#!/bin/bash

# 設定要訓練的參數組合
Nu_values=(22 44)
Nt_values=(1 4)
batch_sizes=(256 512)
learning_rates=(0.001)
# 設定訓練次數
num_runs=3

# 設定其他參數
epoch=800
data="SD"
finetune_model=None
numClasses=4
timeSample=438
C=22
Nc=20
dropoutRate=0.5
weight_decay=0.0001
scheduler=None


# 開始訓練迴圈
for Nu in "${Nu_values[@]}"; do
    for Nt in "${Nt_values[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
        	for lr in "${learning_rates[@]}"; do
                for ((run=1; run<=num_runs; run++)); do
                    echo "Running training with Nu=$Nu, Nt=$Nt, batch_size=$batch_size (Run $run/$num_runs)"
                    python trainer.py --epoch "$epoch" --data "$data" --finetune_model "$finetune_model" --numClasses "$numClasses" --timeSample "$timeSample" --Nu "$Nu" --C "$C" --Nc "$Nc" --Nt "$Nt" --dropoutRate "$dropoutRate" --lr "$lr" --weight_decay "$weight_decay" --scheduler "$scheduler" --batch_size "$batch_size"
                done
            done
        done
    done
done
