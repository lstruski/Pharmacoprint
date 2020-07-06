#!/usr/bin/env bash

dir_root=/mnt/users/struski/local/chemia/data/pharm_fingerprints

act_files=(5HT2A_act_3D_H_onbits 5HT2c_act_3D_H_onbits 5HT6_act_3D_H_onbits catB_act_3D_H_onbits catL_act_3D_H_onbits)
inact_files=(5HT2A_in_3D_H_onbits 5HT2c_in_3D_H_onbits 5HT6_in_3D_H_onbits catB_in_3D_H_onbits catL_in_3D_H_onbits)

for (( i = 0; i < ${#act_files[@]}; ++i )); do
    act_files[i]=${dir_root}/${act_files[i]}
    inact_files[i]=${dir_root}/${inact_files[i]}
done


python supervised_ae_multi_classifiers.py --data_act ${act_files[@]} --data_in ${inact_files[@]} --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/multi_classifers_5_1layer --use_dropout --procedure pre-training_ae training_all

#python supervised_ae_multi_classifiers.py --data_act ${act_files[@]} --data_in ${inact_files[@]} --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 50 20 10 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/multi_classifers_5 --use_dropout --procedure pre-training_ae training_all

# --------------------------------------------------------------

act_files=(5HT2A_act_3D_H_onbits 5HT2c_act_3D_H_onbits 5HT6_act_3D_H_onbits catB_act_3D_H_onbits catL_act_3D_H_onbits D2_act_3D_H_onbits delta_act_3D_H_onbits HIVint_act_3D_H_onbits HIVprot_act_3D_H_onbits HIVrev_act_3D_H_onbits kappa_act_3D_H_onbits mi_act_3D_H_onbits NMDA_act_3D_H_onbits NOP_act_3D_H_onbits NPC1_act_3D_H_onbits)

inact_files=(5HT2A_in_3D_H_onbits 5HT2c_in_3D_H_onbits 5HT6_in_3D_H_onbits catB_in_3D_H_onbits catL_in_3D_H_onbits D2_in_3D_H_onbits delta_in_3D_H_onbits HIVint_in_3D_H_onbits HIVprot_in_3D_H_onbits HIVrev_in_3D_H_onbits kappa_in_3D_H_onbits mi_in_3D_H_onbits NMDA_in_3D_H_onbits NOP_in_3D_H_onbits NPC1_in_3D_H_onbits)

for (( i = 0; i < ${#act_files[@]}; ++i )); do
    act_files[i]=${dir_root}/${act_files[i]}
    inact_files[i]=${dir_root}/${inact_files[i]}
done

python supervised_ae_multi_classifiers.py --data_act ${act_files[@]} --data_in ${inact_files[@]} --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/multi_classifers_all_1layer --use_dropout --procedure pre-training_ae training_all

#python supervised_ae_multi_classifiers.py --data_act ${act_files[@]} --data_in ${inact_files[@]} --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 50 20 10 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/multi_classifers_all --use_dropout --procedure pre-training_ae training_all
