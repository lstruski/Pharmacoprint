#!/usr/bin/env bash

dirs=(data all_data)
targets=(5HT2A 5HT2c 5HT6 D2 HIVint HIVprot HIVrev NMDA NOP NPC1 catB catL kappa mi)

outdir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/ae_other_$(date +%d-%m-%Y)


for d in ${dirs[@]}; do
    for t in ${targets[@]}; do
        if [[ ! -f "${outdir}/${d}/${t}.txt" ]]; then
            echo -e "\033[0;1;31mDIR: ${d}\t \033[0;1;33mTARGET: ${t}\033[0m"
            command="nice -9 taskset -c 46-70 python ae_other.py --data_dir ./data/${d} --name ${t} --pretrain_epochs 75 --dims_layers_ae 7000 4000 500 1000 100 --batch_size 50 --lr 0.0001 --save_dir ${outdir}/${d} --use_dropout --class_weight"

            eval ${command}
            if [[ ! -f "${outdir}/commands_used.txt" ]]; then
                echo ${command} > ${outdir}/commands_used.txt
            fi
        fi
    done
done
