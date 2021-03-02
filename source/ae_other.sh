#!/usr/bin/env bash

gpu=$1

data_dir=./data/pharm_fingerprints/data/3D_H
output_dir=./results/pharm_fingerprints/ae_other/data/3D_H

for filename in $(find ${data_dir} -type f -name "*.npz"); do
  name="${filename##*/}"
  if [[ ! -f ${output_dir}/${name%.*}.txt ]]; then
    CUDA_VISIBLE_DEVICES=${gpu} python3 ae_other.py --filename ${filename} --savefile ${output_dir}/${name%.*}.txt --pretrain_epochs 75 --dims_layers_ae 7000 4000 500 1000 100 --batch_size 50 --lr 0.0001 --use_dropout --class_weight --earlyStopping 10 --use_scheduler
    rm ${output_dir}/models_AE/${name%.*}.pth
  fi
done
