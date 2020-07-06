#!/usr/bin/env bash

#dirs=(data pca_100 all_data_pca_100)
#data_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/other
#output_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/other
#
#results=""
#for d in ${dirs[@]}; do
#    ./results2csv.sh ${data_dir}/${d} ${output_dir} 1 1
#    results="${results} ${output_dir}/${d}_results.csv"
#done
#
#cat ${results} > ${output_dir}_results.csv
#
## ----------------------------------------------------------------
#
#dirs=(data)
#data_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_02-07-2020
#output_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_02-07-2020
#
#results=""
#for d in ${dirs[@]}; do
#    ./results2csv.sh ${data_dir}/${d} ${output_dir} 2
#    results="${results} ${output_dir}/${d}_results.csv"
#done
#
#cat ${results} > ${output_dir}_results.csv
#
## ----------------------------------------------------------------
#
#dirs=(data all_data)
#data_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/ae_other_03-07-2020
#output_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/ae_other_03-07-2020
#
#results=""
#for d in ${dirs[@]}; do
#    ./results2csv.sh ${data_dir}/${d} ${output_dir} 1
#    results="${results} ${output_dir}/${d}_results.csv"
#done
#
#cat ${results} > ${output_dir}_results.csv
## ----------------------------------------------------------------

dirs=(all_data)
data_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_multi_classifiers_06-07-2020
output_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_multi_classifiers_06-07-2020

results=""
for d in ${dirs[@]}; do
    ./results2csv.sh ${data_dir}/${d} ${output_dir} 2
    results="${results} ${output_dir}/${d}_results.csv"
done

cat ${results} > ${output_dir}_results.csv
