#!/usr/bin/env bash

dirs=(all_data)
targets=(5HT2A 5HT2c 5HT6 D2 HIVint HIVprot HIVrev NMDA NOP NPC1 catB catL kappa mi)

outdir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_multi_classifiers_$(date +%d-%m-%Y)

for d in ${dirs[@]}; do
    if [[ ! -d "${outdir}/${d}" ]]; then
        command="nice -9 taskset -c 31-70 python supervised_ae_multi_classifiers.py --data_dir ./data/${d} --name ${targets[@]} --pretrain_epochs 150 --epochs 150 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 --batch_size 50 --lr 0.0001 --save_dir ${outdir}/${d} --use_dropout --procedure pre-training_ae training_classifier --scale 1"

        eval ${command}
        if [[ ! -f "${outdir}/commands_used.txt" ]]; then
            echo ${command} > ${outdir}/commands_used.txt
        fi
    fi
done
