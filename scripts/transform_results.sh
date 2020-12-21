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

#dirs=(all_data)
#data_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_multi_classifiers_06-07-2020
#output_dir=/mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_multi_classifiers_06-07-2020
#
#results=""
#for d in ${dirs[@]}; do
#    ./results2csv.sh ${data_dir}/${d} ${output_dir} 2
#    results="${results} ${output_dir}/${d}_results.csv"
#done
#
#cat ${results} > ${output_dir}_results.csv

# ==========================================================================================

if [[ "$1" == "data" ]]; then
    dirs=(data)
    #supervised_ae_08-07-2020
elif [[ "$1" == "all_data" ]]; then
    dirs=(all_data)
    #supervised_ae_multi_classifiers_08-07-2020
else
    exit 0
fi

data_dir=$2
output_dir=$2


for l in 0.0001 0.00001; do for ep in 100 75 50 25 0; do
add2output=lr_${l}-ep_${ep}

results=""
for d in ${dirs[@]}; do
    ./results2csv.sh ${data_dir}/${add2output}/${d} ${output_dir}/${add2output} 2
    results="${results} ${output_dir}/${add2output}/${d}_results.csv"
done

cat ${results} > ${output_dir}_${add2output}_results.csv
mv ${output_dir}_${add2output}_results.csv $3
done;done

#./transform_results.sh data /mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_08-07-2020 /mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/csv
#./transform_results.sh all_data /mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/supervised_ae_multi_classifiers_08-07-2020 /mnt/users/struski/local/chemia/new_01_07_2020/inne_fingerprinty/csv
#find -type f -name commands_used.txt -exec zip -r results_szerszen.zip {} +;find -type f -name "*.csv" -exec zip -r results_szerszen.zip {} +


#-------------
# diff time
#-------------

# time_start=$(date +"%s")
# ...

# python -c "import sys
# from datetime import timedelta
# print(f'{timedelta(seconds=int(sys.argv[1]))}')
# " $(( $(date +"%s") - ${time_start} ))


