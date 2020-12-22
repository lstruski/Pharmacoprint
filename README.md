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

The data are in directory [data](./data).

# Demo

Directory [scripts](./scripts) contains bash scripts to reproduce the results. Use the following commands:
```bash
bash ae_clr.sh
bash ae_other.sh
bash other
bash supervised_ae.sh
bash supervised_ae_multi_classifiers
```

:warning: Please, in the above files change path to the output directory (variable '*outdir*').

# Tips

If you have any questions about our work, please do not hesitate to contact us by email.
