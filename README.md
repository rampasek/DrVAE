# README #
Dr.VAE: Improving drug response prediction via modeling of drug perturbation effects

Ladislav Rampášek, Daniel Hidru, Petr Smirnov, Benjamin Haibe-Kains and Anna Goldenberg ([Oxford Bioinformatics, 2019](https://doi.org/10.1093/bioinformatics/btz158))

[![DOI](https://zenodo.org/badge/134617808.svg)](https://zenodo.org/badge/latestdoi/134617808) 

### Overview ###

DrVAE, PertVAE and SSVAE(VFAE) implementation in PyTorch

There are several VAE-based models implemented in this repo. All the source is in src/ directory. All models share the same implementation approach and share common implementation of stochastic encoder/decoder blocks.

* layers.py
Implements custom Neural Network layers, e.g. Weight Normalized layer

* blocks.py
Implements several stochastic encoder/decoder blocks as classes that can be instantiated and "wired" together to create VAE-based deep generative graphical model. This implementation is shared across all models.

* utils.py
Contains several functions for data set reading, splitting and then evaluation of results & baselines

* DrVAE.py
Implements Drug Response Variational Autoencoder (Dr.VAE). Here is the implementation of DrVAE class that instantiates necessary encoder/decoder blocks and then provides methods for computation of forward pass, compuation of losses and training/evaluation methods. 

* run\_drvae.py
Main script to run DrVAE. Instantiates DrVAE class instance, trains it and evaluates it against baselines.

* PVAE.py, run\_pvae.py
In analogous way, there is implementations of Perturbation VAE (PertVAE).

* VFAE.py
Implements Variational Fair Autoencoder (VFAE) that can run as Semi-Supervised VAE (SSVAE) as well. VFAE is SSVAE extended by the nuisance variable S. In this file is the implementation of VFAE class that instantiates necessary encoder/decoder blocks and then provides methods for computation of forward pass, compuation of losses and training/evaluation methods. 

* run\_vfae.py
Main script to run VFAE/SSVAE. Instantiates VFAE class instance, trains it and evaluates it against baselines.

### How to run ###

All code is in src/ but it should be run from workspace/ directory. In workspace/ directory, create symbolic links to desired src/ run scripts. Place data files to datafiles/ directory. Then run the scripts.

* Prerequisites:
   Anaconda (the code was tested with Python2.7, briefly with Python3.6 as well)

   PyTorch 0.3.1
```bash
conda install pytorch=0.3.1 -c soumith
```

   rpy2 (OPTIONAL to read RData files, not necessary when running with supplied HDF5 datafile)
```bash
conda install rpy2 
```

* Clone this repo and go to DrVAE/workspace/ directory

* Sample command to run Dr.VAE:
```bash
python run_drvae.py --modelid 'auto' --datafile datafiles/CTRPv2+L1000_FDAdrugs6h_v2.1.h5 --drug 'bortezomib' --stopearly --L 2 --yloss-rate 1 --fold 1 --dim-z1 100 --dim-z3 100 --enc-z1 800 --dec-x 600 --enc-z3 200 --dec-z1 200 --train-w-noise --batch-size 150 --rseed 123
```

* Sample command to run VFAE in SSVAE mode:
```bash
python run_vfae.py --modelid 'auto' --datafile datafiles/CTRPv2+L1000_FDAdrugs6h_v2.1.h5 --drug 'bortezomib' --stopearly --L 2 --yloss-rate 1 --fold 1 --dim-z1 100 --dim-z2 100 --enc-z1 800 --dec-x 600 --enc-z2 200 --dec-z1 200 --alldata --batch-size 150 --rseed 123
```
