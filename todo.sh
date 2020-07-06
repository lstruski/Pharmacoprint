#!/usr/bin/env bash

# active, inactive
for name in 2D_H 2D_noH 3D_H 3D_noH; do
for i in $(find /mnt/users/struski/local/chemia/data/pharm_fingerprints/ -name "*_act_${name}_*"); do
x=${i##*/}
x=${x/_act/}
if [[ ! -f /mnt/users/struski/local/chemia/results/pharm_fingerprints/${name}/${x}.txt ]]; then
nice -9 taskset -c 40-80 python main.py --data_act ${i} --data_in ${i//act/in} --class_weight --logdir /mnt/users/struski/local/chemia/results/pharm_fingerprints/${name}
fi
done
done

# active, zinc
for name in 2D_H 2D_noH 3D_H 3D_noH; do
for i in $(find /mnt/users/struski/local/chemia/data/pharm_fingerprints/ -name "*_act_${name}_*"); do
x=${i##*/}
x=${x/_act/}
if [[ ! -f /mnt/users/struski/local/chemia/results/pharm_fingerprints_zinc/${name}/${x}.txt ]]; then
#ulimit -H -v 4096m  # H - set virtual memory limit to be hard limit, so that process will be killed when exceeding this limit
nice -9 python main.py --data_act ${i} --data_zinc ${i%/*}/randomzinc_${name}_onbits --class_weight --logdir /mnt/users/struski/local/chemia/results/pharm_fingerprints_zinc/${name}
#ulimit -v unlimited
fi
done
done


for i in $(find pharm_fingerprints -type f); do num=$(wc -l < ${i});if [[ ${num} != 18 ]]; then echo ${i} ${num};fi;done

ps --no-headers -u struski -o pcpu,rss | awk '{cpu += $1; rss += $2} END {rss=rss/1024; print cpu, rss,"MB"}'
ps --no-headers -u struski -o pcpu,rss | awk '{cpu += $1; rss += $2} END {rss=rss/(1024^2); print cpu, rss,"GB"}'


# save scores in one file
pca=1
filename=score_results.csv
dir_search="./"
if [[ "${pca}" = "1" ]]; then
echo "case;dataset;method;mean,std;score_type;dim_before_pca;dim_after_pca" > ${filename}
else
echo "case;dataset;method;mean,std;score_type" > ${filename}
fi
for i in $(find ${dir_search} -mindepth 1 -type d); do
name=${i#*/}
echo -n ${name} >> ${filename}
for j in $(find ${i} -mindepth 1 -type f); do
x=${j##*/}
echo -n ";${x::-$((${#name} + 12))}" >> ${filename}
v=0
for k in 1 2 3; do
method=$(sed -n $((${v} + 5))p ${j} | awk '{printf $3}')
if [[ "${pca}" = "1" ]]; then
IFS=', ' read -r -a dim <<< $(tail -n 1 ${j} | awk 'BEGIN {FS=" ";OFS=" ";}{print $4, $6}')
fi
if [[ "${k}" = "1" ]]; then
echo -n ";${method}" >> ${filename}
else
echo -n ";;${method}" >> ${filename}
fi
for p in 1 2 3; do
pv=$((${p} + ${v}))
score_mu=$(sed -n ${pv}p ${j} | awk '{printf $1}')
score_std=$(sed -n ${pv}p ${j} | awk '{printf $2}')
score_type=$(sed -n ${pv}p ${j} | awk '{printf $4}')
if [[ "${p}" = "1" ]]; then
if [[ "${pca}" = "1" ]]; then
echo ";${score_mu};${score_std};${score_type};${dim[0]};${dim[1]}" >> ${filename}
else
echo ";${score_mu};${score_std};${score_type}" >> ${filename}
fi
else
echo ";;;${score_mu};${score_std};${score_type}" >> ${filename}
fi
done
v=$((${v} + 6))
done
done
done
sed -i "s/://g" ${filename}

# results_ae_svm

filename=score_results.csv
dir_search="./"
echo "case;dataset;method;mean,std;score_type" > ${filename}
for i in $(find ${dir_search} -mindepth 1 -maxdepth 1 -type d); do
name=${i#*/}
echo -n ${name} >> ${filename}
for j in $(find ${i} -mindepth 1 -maxdepth 1 -type f); do
x=${j##*/}
echo -n ";${x::-$((${#name} + 12))}" >> ${filename}
v=0
for k in 1 2; do
method=$(sed -n $((${v} + 5))p ${j} | awk '{printf $3}')
if [[ "${k}" = "1" ]]; then
echo -n ";${method}" >> ${filename}
else
echo -n ";;${method}" >> ${filename}
fi
for p in 1 2 3; do
pv=$((${p} + ${v}))
score_mu=$(sed -n ${pv}p ${j} | awk '{printf $1}')
score_std=$(sed -n ${pv}p ${j} | awk '{printf $2}')
score_type=$(sed -n ${pv}p ${j} | awk '{printf $4}')
if [[ "${p}" = "1" ]]; then
echo ";${score_mu};${score_std};${score_type}" >> ${filename}
else
echo ";;;${score_mu};${score_std};${score_type}" >> ${filename}
fi
done
v=$((${v} + 6))
done
done
done
sed -i "s/://g" ${filename}

# results_ae

filename=score_results.csv
dir_search="./"
echo "case;dataset;method;val_score,score_type" > ${filename}
for i in $(find ${dir_search} -mindepth 1 -maxdepth 1 -type d); do
name=${i#*/}
echo -n ${name} >> ${filename}
for j in $(find ${i} -mindepth 1 -maxdepth 1 -type f -name "*.txt"); do
x=${j##*/}
echo -n ";${x::-$((${#name} + 12))};AE_Classif" >> ${filename}
for p in 1 2 3; do
score_mu=$(sed -n ${p}p ${j} | awk '{printf $1}')
score_type=$(sed -n ${p}p ${j} | awk '{printf $NF}')
if [[ "${p}" = "1" ]]; then
echo ";${score_mu};${score_type}" >> ${filename}
else
echo ";;;${score_mu};${score_type}" >> ${filename}
fi
done
done
done
sed -i "s/://g" ${filename}


# --------------------------------------------------------------

# active, inactive
for name in 2D_H 2D_noH 3D_H 3D_noH; do
for i in $(find /mnt/users/struski/local/chemia/data/pharm_fingerprints/ -name "*_act_${name}_*"); do
x=${i##*/}
x=${x/_act/}
if [[ ! -f /mnt/users/struski/local/chemia/results_ae_svm/pharm_fingerprints/${name}/${x}.txt ]]; then
nice -9 taskset -c 24-39 python ae_svm.py --data_act ${i} --data_in ${i//act/in} --pretrain_epochs 75 --dims_layers_ae 7000 4000 500 1000 100 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/results_ae_svm/pharm_fingerprints/${name} --use_dropout --class_weight
fi
done
done

for name in 2D_H 2D_noH 3D_H 3D_noH; do
for i in $(find /mnt/users/struski/local/chemia/data/pharm_fingerprints/ -name "*_act_${name}_*"); do
x=${i##*/}
x=${x/_act/}
if [[ ! -f /mnt/users/struski/local/chemia/results_ae/pharm_fingerprints/${name}/${x}.txt ]]; then
nice -9 taskset -c 24-39 python supervised_ae.py --data_act ${i} --data_in ${i//act/in} --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 50 20 10 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/results_ae/pharm_fingerprints/${name} --use_dropout --procedure pre-training_ae training_classifier --scale_loss 1
fi
done
done

# active, zinc
for name in 2D_H 2D_noH 3D_H 3D_noH; do
for i in $(find /mnt/users/struski/local/chemia/data/pharm_fingerprints/ -name "*_act_${name}_*"); do
x=${i##*/}
x=${x/_act/}
if [[ ! -f /mnt/users/struski/local/chemia/results_ae_svm/pharm_fingerprints_zinc/${name}/${x}.txt ]]; then
nice -9 taskset -c 24-39 python ae_svm.py --data_act ${i} --data_zinc ${i%/*}/randomzinc_${name}_onbits --scale_zinc 9 --pretrain_epochs 75 --dims_layers_ae 7000 4000 500 1000 100 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/results_ae_svm/pharm_fingerprints_zinc/${name} --use_dropout --class_weight
fi
done
done

for name in 2D_H 2D_noH 3D_H 3D_noH; do
for i in $(find /mnt/users/struski/local/chemia/data/pharm_fingerprints/ -name "*_act_${name}_*"); do
x=${i##*/}
x=${x/_act/}
if [[ ! -f /mnt/users/struski/local/chemia/results_ae/pharm_fingerprints_zinc/${name}/${x}.txt ]]; then
nice -9 taskset -c 24-39 python supervised_ae.py --data_act ${i} --data_zinc ${i%/*}/randomzinc_${name}_onbits --scale_zinc 9 --pretrain_epochs 75 --epochs 100 --dims_layers_ae 7000 4000 500 1000 100 --dims_layers_classifier 100 50 20 10 --batch_size 50 --lr 0.0001 --dir_save /mnt/users/struski/local/chemia/results_ae/pharm_fingerprints_zinc/${name} --use_dropout --procedure pre-training_ae training_classifier --scale_loss 1
fi
done
done

############################################################
# inne_fingerprinty

# active, inactive
outdir=/mnt/users/struski/local/chemia/results/inne_fingerprinty
for i in $(find /mnt/users/struski/local/chemia/data/inne_fingerprinty/ -name "*_act_*"); do
x=${i##*/}
x=${x//_act/}
if [[ -f ${i//act/in} ]] && [[ ! -f "${outdir}/${x//csv/txt}" ]]; then
nice -9 taskset -c 15-45 python main.py --data_act ${i} --data_in ${i//act/in} --class_weight --logdir ${outdir}
fi
done

# active, zinc
outdir=/mnt/users/struski/local/chemia/results/inne_fingerprinty_zinc
for i in $(find /mnt/users/struski/local/chemia/data/inne_fingerprinty/ -name "*_act_*"); do
x=${i##*/}
x=${x//_act/}
if [[ -f ${i%/*}/randomzinc${i##*_act} ]] && [[ ! -f "${outdir}/${x//csv/txt}" ]]; then
nice -9 taskset -c 15-45 python main.py --data_act "$i" --data_zinc "${i%/*}"/randomzinc"${i##*_act}" --class_weight --logdir ${outdir} --dense
fi
done

outdir=/mnt/users/struski/local/chemia/results/pharm_fingerprints_zinc
for i in $(find /mnt/users/struski/local/chemia/data/pharm_fingerprints/ -name "*_act_*"); do
x=${i##*/}
x=${x//_act/}
if [[ -f ${i%/*}/randomzinc${i##*_act} ]] && [[ ! -f "${outdir}/${x//csv/txt}" ]]; then
nice -9 taskset -c 15-45 python main.py --data_act "$i" --data_zinc "${i%/*}"/randomzinc"${i##*_act}" --class_weight --logdir ${outdir}
fi
done

