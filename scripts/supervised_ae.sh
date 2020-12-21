#!/usr/bin/env bash

dirs=(data)
targets=(5HT2A 5HT2c 5HT6 D2 HIVint HIVprot HIVrev NMDA NOP NPC1 catB catL kappa mi)

if [[ "$1" == "" ]]; then
    current_date=$(date +%d-%m-%Y)
else
    current_date=$1
fi

# change path!!!
outdir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_${current_date}

ae_model="7000 4000 500 1000 100"
classifier_model=("100" "100-25" "100-50-20-10")

if [[ "$2" == "" ]]; then

    for sc in 1 1.5 2 4; do for lr in 0.0001 0.00001; do for ep in 100 75 50 25 0; do for clr_model in ${classifier_model[@]}; do
        add2output=lr_${lr}-pretrnEpoch_${ep}-scale_${sc}-clr_${clr_model}
        proc="pre-training_ae training_all"
        if [[ "${ep}" == "0" ]]; then
            proc="training_all"
        fi

        for d in ${dirs[@]}; do
            for t in ${targets[@]}; do
                if [[ ! -f "${outdir}/${add2output}/${d}/${t}.txt" ]]; then
                    echo -e "\033[0;1;31mDIR: ${d}\t \033[0;1;33mTARGET: ${t}\033[0m"

                    command="python ../source/supervised_ae.py --data_dir ../data/${d} --name ${t} --pretrain_epochs ${ep} --epochs 150 --dims_layers_ae ${ae_model} --dims_layers_classifier ${clr_model//-/ } --batch_size 50 --lr ${lr} --save_dir ${outdir}/${add2output}/${d} --use_dropout --procedure ${proc} --scale_loss ${sc}"

                    echo -e "\033[1;31mRUN: \033[0;1;32m${command}\033[0m"
                    eval ${command}
                    if [[ ! -f "${outdir}/${add2output}/commands_used.txt" ]]; then
                        echo ${command} > ${outdir}/${add2output}/commands_used.txt
                    fi
                fi
            done

            ./results2csv.sh ${outdir}/${add2output}/${d} ${outdir}/${add2output} 2
        done

    done;done;done;done

else

    exit

fi
