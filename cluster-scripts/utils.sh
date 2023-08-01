#!/bin/bash

source ./test.config

function killOldProcesses() {
    local PORT_NO=$1
    local PS_NAME="[f]ed_avg.py"
    kill -9 $(ps -ef|grep "$PS_NAME"|grep "${PORT_NO}"|awk '{ print $2 }')
}

function cleanOutput() {
    rm -f $PWD/../federated-learning/result_*.txt
    rm -f $PWD/../federated-learning/model_*.pt
}

function clean() {
    local PORT_NO=$1
    killOldProcesses "${PORT_NO}"
    cleanOutput
}

function arrangeOutput(){
    local DIR_NAME=$1
    # gather outputs
    rm -rf output/
    mkdir -p output/
    cp $PWD/../federated-learning/result_*.txt output/
    cp $PWD/../federated-learning/model_*.pt output/
    cp $PWD/../server.log output/

    mkdir -p "${DIR_NAME}"
    mv output/ "${DIR_NAME}"
}


function testFinish() {
    local PORT_NO=$1
    local PS_NAME="[f]ed_avg.py"
    while : ; do
        echo "[`date`] sleep 60 seconds before detecting process status"
        sleep 60
        local count=$(ps -ef|grep "${PS_NAME}"|grep "${PORT_NO}"|wc -l)
        if [[ $count -eq 0 ]]; then
            break
        fi
        echo "[`date`] process still active"
    done
}

