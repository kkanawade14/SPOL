#!/bin/bash

mkdir -p logs

declare -i dim=8
declare -i nb_train=5000
declare -i level=3

keys=(logKL aff_S3 aff_F9)

# --- Full DS run ---
echo "===== FULL DATASET GENERATION (level $level, nb_train $nb_train) ====="
for key in "${keys[@]}"; do
    echo "--- Generating $key ---"
    python -m data_generation.boussinesq.generate --dim $dim --key $key --level $level --nb_train $nb_train --seed 42 > logs/DS_BSNQ_d${dim}_${key}_level${level}_ntrain${nb_train}.log 2>&1
    echo "--- $key generation done ---"
done

