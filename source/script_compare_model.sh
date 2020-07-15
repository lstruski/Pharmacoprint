#!/usr/bin/env bash

# How to use it: ./script_compare_model.sh ./supervised_ae_multi_classifiers_10-07-2020
dir=$1

if [[ "${dir: -1}" == "/" ]]; then
  dir=${dir:: -1}
fi


num=($(echo ${dir} | tr "/" "\n"))
num=$(( ${#num[@]} - 1))
#echo $num

if [[ -f ${dir}/*results.csv ]]; then
    rm ${dir}/*results.csv
fi

files=()
for i in $(find ${dir} -mindepth 2 -type f -name "*.csv"); do
    files+=( ${i} )
done

#echo ${files[@]}

python -c "import argparse
from pathlib import PurePosixPath
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', help='Files *.csv')
args = parser.parse_args()


for i in range(len(args.data)):
    df = pd.read_csv(args.data[i], delimiter=';')
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(method='ffill', axis=0, inplace=True)
    df = df[['case', 'dataset', 'method', 'score_type', 'mean']]
    df.rename(columns = {'mean': PurePosixPath(args.data[i]).parents[0].name}, inplace=True)
    if i == 0:
        data = df.copy()
    else:
        data = pd.merge(data, df, how='inner', on=['case', 'dataset', 'method', 'score_type'])
#print(data)
data.to_csv(f'{PurePosixPath(args.data[0]).parents[${num}]}_results.csv', index=False, sep=';')

size = len(args.data)
colors = {'ACC': 'red', 'ROC_AUC': 'green', 'MCC': 'blue'}

df = data.loc[:, data.columns[:-size]]
df['mean'] = data.loc[:, data.columns[-size:]].max(axis=1)
df['idxmax'] = data[data.columns[-size:]].idxmax(axis=1)
df.to_csv(f'{PurePosixPath(args.data[0]).parents[${num}]}/{PurePosixPath(args.data[0]).name}', index=False, sep=';')
del df

data.drop(columns=['case', 'method'], inplace=True)
#print(data)

grouped = data.groupby(['dataset'])
#print(grouped.groups.keys())

for title, group in grouped:
    fig = plt.figure(figsize=(len(args.data) // 2, 6))
    plt.title(f'{title}', fontsize=15, fontname='serif', fontweight='bold')

    values = group[group.columns[-size:]].to_numpy()
    names = group.columns[-size:].to_list()
    name_score = group.score_type.values
    for i in range(len(name_score)):
        plt.plot(range(size), values[i, :], label=f'{name_score[i]}', color=colors[name_score[i]])    
    
    min_x, max_x, min_y, max_y = plt.axis()

    plt.xticks(range(size), names, rotation=40, ha='right', fontsize=10, fontname='serif')
    plt.xlabel('models', fontsize=15, fontname='serif')
    plt.ylabel('score', fontsize=15, fontname='serif')
    plt.legend()
    plt.savefig(f'{PurePosixPath(args.data[0]).parents[${num}]}/{title}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
    #plt.show()
    plt.close()
" --data ${files[@]}

