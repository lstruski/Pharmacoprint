#!/usr/bin/env bash

data_dir=./data/inne_fingerprinty/data
output_dir=./results/inne_fingerprinty/other/data

for filename in $(find ${data_dir} -type f -name "*.npz"); do
  name="${filename##*/}"
  python3 other.py --filename ${filename} --savefile ${output_dir}/${name%.*}.txt --class_weight
done

# ================================================

data_dir=./data/pharm_fingerprints/data/
output_dir=./results/pharm_fingerprints/other/data

for filename in $(find ${data_dir} -type f -name "*.npz"); do
  name="${filename##*/}"
  outdir=${filename%/*}
  if [[ ! -f ${output_dir}/${outdir##*/}/${name%.*}.txt ]]; then
    python3 other.py --filename ${filename} --savefile ${output_dir}/${outdir##*/}/${name%.*}.txt --class_weight
  fi
done

# ================================================

data_dir=./data/pharm_fingerprints/pca_0.9/
output_dir=./results/pharm_fingerprints/other/pca_0.9

#data_dir=./data/pharm_fingerprints/pca_100/
#output_dir=./results/pharm_fingerprints/other/pca_100

for filename in $(find ${data_dir} -type f -name "*.npz"); do
  name="${filename##*/}"
  outdir=${filename%/*}
  if [[ ! -f ${output_dir}/${outdir##*/}/${name%.*}.txt ]]; then
    python3 other.py --filename ${filename} --savefile ${output_dir}/${outdir##*/}/${name%.*}.txt --class_weight
  fi
done
