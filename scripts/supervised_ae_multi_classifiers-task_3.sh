#!/usr/bin/env bash

dirs=(all_data)
targets=(5HT2A 5HT2c 5HT6 D2 HIVint HIVprot HIVrev NMDA NOP NPC1 catB catL kappa mi)

if [[ "$1" == "" ]]; then
    current_date=$(date +%d-%m-%Y)
else
    current_date=$1
fi
outdir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_multi_classifiers-task3_${current_date}

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
            for idt in ${!targets[@]}; do

                if [[ ! -d "${outdir}/${add2output}-target_${targets[${idt}]}/${d}" ]]; then

                    command="python ../source/supervised_ae_multi_classifiers.py --data_dir ../data/${d} --name ${targets[@]::${idt}} ${targets[@]:$(( ${idt} + 1))} --pretrain_epochs ${ep} --epochs 150 --dims_layers_ae ${ae_model} --dims_layers_classifier ${clr_model//-/ } --batch_size 50 --lr ${lr} --save_dir ${outdir}/${add2output}-target_${targets[${idt}]}/${d} --use_dropout --procedure ${proc} --scale ${sc} --save_model ae"

                    echo -e "\033[1;31mRUN: \033[0;1;32m${command}\033[0m"
                    eval ${command}
                    if [[ ! -f "${outdir}/${add2output}-target_${targets[${idt}]}/commands_used.txt" ]]; then
                        echo ${command} > ${outdir}/${add2output}-target_${targets[${idt}]}/commands_used.txt
                    fi

                    command="python ../source/supervised_ae_multi_classifiers.py --data_dir ../data/${d} --name ${targets[${idt}]} --epochs 150 --dims_layers_ae ${ae_model} --dims_layers_classifier ${clr_model//-/ } --batch_size 50 --lr ${lr} --save_dir ${outdir}/${add2output}-target_${targets[${idt}]}/${d}_res --use_dropout --procedure training_classifier --scale 1 --ae ${outdir}/${add2output}-target_${targets[${idt}]}/${d}/ae_model.pth"

                    echo -e "\033[1;31mRUN: \033[0;1;34m${command}\033[0m"
                    eval ${command}
                    if [[ ! -f "${outdir}/${add2output}-target_${targets[${idt}]}/commands_used-next.txt" ]]; then
                        echo ${command} > ${outdir}/${add2output}-target_${targets[${idt}]}/commands_used-next.txt
                    fi
                fi

                ./results2csv.sh ${outdir}/${add2output}-target_${targets[${idt}]}/${d} ${outdir}/${add2output}-target_${targets[${idt}]} 2

            done

        done

    done;done;done;done

else

    exit

fi