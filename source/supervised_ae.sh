#!/usr/bin/env bash

gpu=0 # GPU device number

data_dir=./data/pharm_fingerprints/data/3D_H
output_dir=./results/pharm_fingerprints/supervised_ae/data/3D_H

num_split=3
ae_model="7000 4000 500 1000 100"
classifier_model=("100" "100-25" "100-50-20-10")

for filename in $(find ${data_dir} -maxdepth 1 -type f -name "*.npz"); do
  name="${filename##*/}"

  if [[ ! -d ${output_dir}/${name%.*}_splits ]]; then
    python3 data_splits.py --filename ${filename} --save_dir ${output_dir}/${name%.*}_splits --n_splits ${num_split}
  fi
  for sc in 1 1.5 2 4; do for lr in 0.001 0.0001; do for ep in 50 0; do for clr_model in ${classifier_model[@]}; do
    add2output=lr_${lr}-pretrnEpoch_${ep}-scale_${sc}-clr_${clr_model}
    echo -e "\033[0;31m${name%.*}: $add2output\033[0m"
    proc="pre-training_ae training_all"
    if [[ "${ep}" == "0" ]]; then
        proc="training_all"
    fi

    if [[ ! -d ${output_dir}/${name%.*}/${add2output} ]]; then
      for i in $(seq 0 $(( $num_split - 1 ))); do
        CUDA_VISIBLE_DEVICES=${gpu} python3 supervised_ae_single.py --filename ${output_dir}/${name%.*}_splits/${i}.npz --pretrain_epochs ${ep} --epochs 150 --dims_layers_ae ${ae_model} --dims_layers_classifier ${clr_model//-/ } --batch_size 50 --lr ${lr} --save_dir ${output_dir}/${name%.*}/${add2output} --use_dropout --procedure ${proc} --scale_loss ${sc} --earlyStopping 10 --use_scheduler
        rm ${output_dir}/${name%.*}/${add2output}/models_AE/${name%.*}_splits_${i}.pth
      done
    fi
  done;done;done;done

  rm -r ${output_dir}/${name%.*}_splits
done
