#!/bin/bash

mkdir -p logs

declare -i dim=8
declare -i level=3
declare -i seed=42
declare -i total_trials=10

experiment="nsb"

keys=(logKL aff_S3 aff_F9)
variables=(p u)
m_schedule=(100 200 300 400 500)

echo "===== PDR RUNS: experiment=$experiment dim=$dim level=$level ====="

for variable in "${variables[@]}"; do

    # Choose norm based on variable
    if [ "$variable" = "p" ]; then
        norm="l2"
    elif [ "$variable" = "u" ]; then
        norm="l4"
    else
        echo "Unknown variable: $variable"
        exit 1
    fi

    echo "===== Variable=$variable | Norm=$norm ====="

    for key in "${keys[@]}"; do
        echo "--- Running PDR: variable=$variable norm=$norm key=$key ---"

        python -m pdr.run_pdr \
            --experiment "$experiment" \
            --variable "$variable" \
            --key "$key" \
            --dim "$dim" \
            --level "$level" \
            --norm "$norm" \
            --m_schedule "${m_schedule[@]}" \
            --seed "$seed" \
            --total_trials "$total_trials" \
            > logs/PDR_${experiment}_${variable}_${norm}_d${dim}_${key}_level${level}.log 2>&1

        echo "--- Done: variable=$variable key=$key ---"
    done
done

echo "===== ALL PDR RUNS DONE ====="