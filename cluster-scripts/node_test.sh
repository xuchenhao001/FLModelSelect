#!/bin/bash

# set -x

source ./test.config
source ./utils.sh

function main() {
    for i in "${!MODEL_DS[@]}"; do
        model_ds=(${MODEL_DS[i]//-/ })
        model=${model_ds[0]}
        dataset=${model_ds[1]}
        echo "[`date`] ALL_NODE_TEST UNDER: ${model} - ${dataset}"

        for i in "${!SCHEMES[@]}"; do
            scheme="${SCHEMES[i]}"
            # run the test for each scheme
            run ${scheme} ${model} ${dataset} 0.3
            run ${scheme} ${model} ${dataset} 0.5
            run ${scheme} ${model} ${dataset} 0.7
        done
    done
}

function run() {
    local scheme_name=$1
    local model_name=$2
    local dataset_name=$3
    local frac=$4
    local noniid_zipf_s=2.45
    output_dir="all_test_results/frac-${frac}/${model_name}-${dataset_name}/${scheme_name}"

    if [[ ! -d "${output_dir}" ]]; then
        echo "[`date`] ## ${output_dir} start ##"
        clean "${FL_LISTEN_PORT}"
        PYTHON_CMD="python3 -u fed_avg.py --gpu=${GPU_NO} --fl_listen_port=${FL_LISTEN_PORT} --dataset_size_variant --scheme=${scheme_name} --model=${model_name} --dataset=${dataset_name}  --noniid_zipf_s=${noniid_zipf_s} --frac=${frac}"
        cd $PWD/../federated-learning/; $PYTHON_CMD > $PWD/../server.log 2>&1 &
        cd -
        # detect test finish or not
        testFinish "${FL_LISTEN_PORT}"
        # gather output, move to the right directory
        arrangeOutput "${output_dir}"
        echo "[`date`] ## ${output_dir} done ##"
    fi
}

FL_LISTEN_PORT=$1
if [[ -z "${FL_LISTEN_PORT}" ]]; then
    FL_LISTEN_PORT="8800"
fi
GPU_NO=$2
if [[ -z "${GPU_NO}" ]]; then
    GPU_NO="-1"
fi
main > test.log 2>&1 &
