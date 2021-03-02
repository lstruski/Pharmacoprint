# Pharmacoprint

Python code reproducing the results reported in the paper:

D. Warszycki, Ł. Struski, M. Śmieja, R. Kafel, R. Kurczab, "Pharmacoprint - a new approach in computer-aided drug design combining pharmacophore fingerprint and artificial intelligence", 2020

# Requirements

Please, install the following packages:
* numpy
* scipy
* sklearn
* tensorboardX
* pytorch 1.6
* tqdm

# Data

We share the data [here](https://ujchmura-my.sharepoint.com/:f:/g/personal/lukasz_struski_uj_edu_pl/Egao2rcXN8hGnTJtNH5cs88BsHzos__xiDA_Z-vsUvajwg?e=F7Sqer), since their size is too big to share on Github.

# Run

In the beginning, you should download data from [here](https://ujchmura-my.sharepoint.com/:f:/g/personal/lukasz_struski_uj_edu_pl/Egao2rcXN8hGnTJtNH5cs88BsHzos__xiDA_Z-vsUvajwg?e=F7Sqer) and put it in the directory [source](./source).

Directory [source](./source) contains bash scripts to reproduce the results. Use the following commands:
```bash
bash other.sh
bash ae_other.sh
bash supervised_ae.sh
```

:warning: Please, in the above files change path to the output directory (variable '*outdir*'). Default: results will be saved in the directory 'results'. 

You can view plots of how the model learns by typing the following command in the console:
```bash
tensorboard --logdir ./results
```

# Results

We have placed the individual results of our experiments in the folder [results](./results)


# Tips

If you have any questions about our work, please do not hesitate to contact us by email.
