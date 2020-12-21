#!/usr/bin/env bash


dirs=(data pca_100 all_data_pca_100)
targets=(5HT2A 5HT2c 5HT6 D2 HIVint HIVprot HIVrev NMDA NOP NPC1 catB catL kappa mi)

# change path!!!
outdir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/other

for d in ${dirs[@]}; do
    for t in ${targets[@]}; do
        echo -e "\033[0;1;31mDIR: ${d}\t \033[0;1;33mTARGET: ${t}\033[0m"
        if [[ ! -f "${outdir}/${d}/${t}.txt" ]]; then
            python ../source/other.py --data_dir ../data/${d} --name ${t} --class_weight --logdir ${outdir}/${d}
        fi

        if [[ ! -f "${outdir}/${d}/nn/${t}.txt" ]]; then
            command="python ../source/supervised_ae.py --data_dir ./data/${d} --name ${t} --epochs 100 --dims_layers_classifier -1 500 100 50 10 --batch_size 50 --lr 0.0001 --save_dir ${outdir}/${d}/nn --use_dropout --procedure training_classifier --scale_loss 1"

            eval ${command}
            if [[ ! -f "${outdir}/commands_used.txt" ]]; then
                echo ${command} > ${outdir}/commands_used.txt
            fi
        fi
    done
done
