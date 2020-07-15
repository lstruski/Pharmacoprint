#!/usr/bin/env bash

# How to use it: ./script_compare_scores.sh

dirs=(ae_02-07-2020 ae_multi_classifiers_06-07-2020 ae_other_03-07-2020 other supervised_ae_08-07-2020 supervised_ae_10-07-2020 supervised_ae_multi_classifiers_08-07-2020 supervised_ae_multi_classifiers_10-07-2020)

files=()
for d in ${dirs[@]}; do
    for i in $(find ${d} -mindepth 1 -maxdepth 1 -type f -name "*.csv"); do
        files+=( ${i} )
    done
done
#echo ${files[@]}

if [[ ! -d "./compare_scores" ]]; then
    mkdir ./compare_scores
fi

python -c "import argparse
from pathlib import PurePosixPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', help='Files *.csv')
args = parser.parse_args()

data = [] 
for i in range(len(args.data)):
    data.append(pd.read_csv(args.data[i], delimiter=';'))
    data[i].dropna(axis=1, how='all', inplace=True)
    data[i].fillna(method='ffill', axis=0, inplace=True)
    data[i]['from_file'] = PurePosixPath(args.data[i]).parents[0].name

data = pd.concat(data, ignore_index=True)
#print(data)
data.sort_values(by=['dataset', 'mean'], ascending=[True, False]).to_csv('./compare_scores/results.csv', index=False, sep=';')

grouped = data.groupby(['dataset', 'score_type'])
#print(grouped.groups.keys())

#for name, group in grouped:
#    print(f'Dataset: {name[0]}, score: {name[1]}')
#    print(group)

datasets = data.dataset.unique()
scores = data.score_type.unique()

colors = {'ACC': 'red', 'ROC_AUC': 'green', 'MCC': 'blue'}

for title in datasets:
    fig = plt.figure(figsize=(10, 6))
    plt.title(f'{title}', fontsize=15, fontname='serif', fontweight='bold')
    names = None
    for s in scores:
        df = grouped.get_group((title, s))
        #print(df)
        plt.plot(range(df.shape[0]), df['mean'].to_numpy(), label=f'{s}', color=colors[s])
        if names is None:
            names = df[['from_file', 'method', 'case']].apply(lambda x: ' | '.join(x), axis=1).values
        else:
            assert len(set(names).intersection(df[['from_file', 'method', 'case']].apply(lambda x: ' | '.join(x), axis=1).values)) == len(names)
    
    min_x, max_x, min_y, max_y = plt.axis()
    
    tmp = [f'{idx} -- {val}' for idx, val in enumerate(names)]
    num_letters = np.max([len(a) for a in tmp])
    tmp = [ f'{a:<{num_letters}}  {b:<{num_letters}}' for a, b in zip(tmp[::2], tmp[1::2])]
    
    #plt.xticks(range(df.shape[0]), names, rotation=40, ha='right')
    plt.xticks(range(df.shape[0]), range(df.shape[0]), rotation=0, ha='right')
    plt.xlabel('\n'.join(tmp), ha='center', va='center', color='k', labelpad=75, fontsize=9, fontname='serif')    
    #plt.xlabel('models', fontsize=15, fontname='serif', fontweight='bold')
    plt.ylabel('score', fontsize=15, fontname='serif')
    plt.yticks(np.linspace(min_y, max_y, 15, endpoint=True))
    plt.legend()
    plt.savefig(f'./compare_scores/{title}.png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
    #plt.show()
    plt.close()
" --data ${files[@]}

