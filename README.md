# Pharmacoprint

The Python code reproducing the results reported in the paper:

D. Warszycki, Ł. Struski, M. Śmieja, R. Kafel, R. Kurczab, "Pharmacoprint - a new approach in computer-aided drug design combining pharmacophore fingerprint and artificial intelligence", preprint, 2020

# Requirements

Please, install the following packages:
* numpy
* scipy
* sklearn
* tensorboardX
* pytorch 1.6
* tqdm

# Data

The input data can be downloaded from [here](http://ww2.ii.uj.edu.pl/~struski/results/Pharmacoprint/data.zip) because their size is too big to share them on Github.

# Run

To run the code, you should download the [data](http://ww2.ii.uj.edu.pl/~struski/results/Pharmacoprint/data.zip) directory and put it in the directory [source](./source).

The directory [source](./source) contains bash scripts to reproduce our results. Use the following commands to run subsequent models:
```bash
bash other.sh         #classical machine learninig models such as SVM, logistic regression, PCA, etc.
bash ae_other.sh      #classifiers, which use auto-encoder reduction
bash supervised_ae.sh #supervised auto-encoder classifiers
```

:warning: Please, in the above files change the path to the output directory (variable '*outdir*'). Default: results will be saved in the directory 'results'. 

You can view the training curves of the model by typing the following command in the console:
```bash
tensorboard --logdir ./results
```

# Results

Detailed results of the experiments reported in the paper can be found in [results](./results)


# Tips

If you have any questions about our work, please do not hesitate to contact us by email.
