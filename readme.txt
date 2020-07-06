filesave=results_2019-10-06_pca_balance_weight.txt
if [ -f ${filesave} ]; then rm ${filesave}; fi
i=1
files=($(find ./on_bits/ -type f -name "*act*"))
for filename in ${files[*]}; do
echo -en "\r[${i}|${#files[*]}]"
echo "---------------------------------------------------------------------" >> ${filesave}
python main.py --data_act ${filename} --data_in ${filename/_act_/_in_} --class_weight --pca_dim 100 >> ${filesave}
i=$((i + 1))
done
echo 



filesave=results_2019-10-06_pca_not_balance_weight.txt
if [ -f ${filesave} ]; then rm ${filesave}; fi
i=1
files=($(find ./on_bits/ -type f -name "*act*"))
for filename in ${files[*]}; do
echo -en "\r[${i}|${#files[*]}]"
echo "---------------------------------------------------------------------" >> ${filesave}
python main.py --data_act ${filename} --data_in ${filename/_act_/_in_} --pca_dim 100 >> ${filesave}
i=$((i + 1))
done
echo 



filesave=results_2019-10-06_pca_zinc_not_balance_weight.txt
if [ -f ${filesave} ]; then rm ${filesave}; fi
i=1
files=($(find ./on_bits/ -type f -name "*act*"))
for filename in ${files[*]}; do
echo -en "\r[${i}|${#files[*]}]"
echo "---------------------------------------------------------------------" >> ${filesave}
python main.py --data_act ${filename} --pca_dim 100 >> ${filesave}
i=$((i + 1))
done
echo



filesave=results_2019-10-06_pca_zinc_balance_weight.txt
if [ -f ${filesave} ]; then rm ${filesave}; fi
i=1
files=($(find ./on_bits/ -type f -name "*act*"))
for filename in ${files[*]}; do
echo -en "\r[${i}|${#files[*]}]"
echo "---------------------------------------------------------------------" >> ${filesave}
python main.py --data_act ${filename} --class_weight --scale_zinc 9 --pca_dim 100 >> ${filesave}
i=$((i + 1))
done
echo

########################################################################################

filesave=results_2019-10-26_net_balance_data.txt
if [ -f ${filesave} ]; then rm ${filesave}; fi
i=1
files=($(find ./on_bits/ -type f -name "*act*"))
for filename in ${files[*]}; do
echo -en "\r[${i}|${#files[*]}]"
echo "---------------------------------------------------------------------" >> ${filesave}
tmp=${filename##*/}
echo -n "${tmp%%_*} " >> ${filesave}
python supervised_ae.py --data_act ${filename} --data_in ${filename/_act_/_in_} --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 50 20 10 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/act_in/${tmp%%_*} --use_dropout --logger ${filesave} --procedure pre-training_ae training_all --scale_loss 1
i=$((i + 1))
done
echo

filesave=results_2019-10-26_net_zinc_balance_data.txt
if [ -f ${filesave} ]; then rm ${filesave}; fi
i=1
files=($(find ./on_bits/ -type f -name "*act*"))
for filename in ${files[*]}; do
echo -en "\r[${i}|${#files[*]}]"
echo "---------------------------------------------------------------------" >> ${filesave}
tmp=${filename##*/}
echo -n "${tmp%%_*} " >> ${filesave}
python supervised_ae.py --data_act ${filename} --scale_zinc 1 --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 50 20 10 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/act_zinc/${tmp%%_*} --use_dropout --logger ${filesave} --procedure pre-training_ae training_all --scale_loss 1
i=$((i + 1))
done
echo
