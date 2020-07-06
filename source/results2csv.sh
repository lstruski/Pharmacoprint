#!/usr/bin/env bash

dir_search=$1

name=${dir_search##*/}
filename=$2/${name}_results.csv
case=$3
use_nn=$4

targets=(5HT2A 5HT2c 5HT6 D2 HIVint HIVprot HIVrev NMDA NOP NPC1 catB catL kappa mi)

echo "case;dataset;method;mean;std;score_type" > ${filename}
echo -n ${name} >> ${filename}
for t in ${targets[@]}; do
    echo -n ";${t}" >> ${filename}

    if [[ "${case}" == "1" ]]; then
        v=0
        for k in 1 2 3; do
            method=$(sed -n $((${v} + 5))p ${dir_search}/${t}.txt | awk '{printf $3}')
            if [[ "${k}" = "1" ]]; then
                echo -n ";${method}" >> ${filename}
            else
                echo -n ";;${method}" >> ${filename}
            fi
            for p in 1 2 3; do
                pv=$((${p} + ${v}))
                score_mu=$(sed -n ${pv}p ${dir_search}/${t}.txt | awk '{printf $1}')
                score_std=$(sed -n ${pv}p ${dir_search}/${t}.txt | awk '{printf $2}')
                score_type=$(sed -n ${pv}p ${dir_search}/${t}.txt | awk '{printf $4}')
                if [[ "${p}" = "1" ]]; then
                    echo ";${score_mu};${score_std};${score_type}" >> ${filename}
                else
                    echo ";;;${score_mu};${score_std};${score_type}" >> ${filename}
                fi
            done
            v=$((${v} + 6))
        done
        if [[ "${use_nn}" == "1" ]]; then
            for p in 1 2 3; do
                score_mu=$(sed -n ${p}p ${dir_search}/nn/${t}.txt | awk '{printf $1}')
                score_std=""
                score_type=$(sed -n ${p}p ${dir_search}/nn/${t}.txt | awk '{printf $NF}')
                if [[ "${p}" = "1" ]]; then
                    echo ";;neural network;${score_mu};${score_std};${score_type}" >> ${filename}
                else
                    echo ";;;${score_mu};${score_std};${score_type}" >> ${filename}
                fi
            done
        fi
    elif [[ "${case}" == "2" ]]; then
        for p in 1 2 3; do
            score_mu=$(sed -n ${p}p ${dir_search}/${t}.txt | awk '{printf $1}')
            score_std=""
            score_type=$(sed -n ${p}p ${dir_search}/${t}.txt | awk '{printf $NF}')
            if [[ "${p}" = "1" ]]; then
                echo ";neural network;${score_mu};${score_std};${score_type}" >> ${filename}
            else
                echo ";;;${score_mu};${score_std};${score_type}" >> ${filename}
            fi
        done
    fi
done
