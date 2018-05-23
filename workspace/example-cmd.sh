### sample Dr.VAE run commands
# expected to be run from DrVAE/workspace/ directory

python run_drvae.py --modelid 'auto' --datafile datafiles/CTRPv2+L1000_FDAdrugs6h_v2.1.h5 --drug 'bortezomib' --stopearly --L 2 --yloss-rate 1 --fold 1 --dim-z1 100 --dim-z3 100 --enc-z1 800 --dec-x 600 --enc-z3 200 --dec-z1 200 --train-w-noise --batch-size 150 --rseed 123

python run_drvae.py --modelid 'auto' --datafile datafiles/CTRPv2+L1000_FDAdrugs6h_v2.1.h5 --drug 'vorinostat' --stopearly --L 2 --yloss-rate 1 --fold 1 --dim-z1 100 --dim-z3 100 --enc-z1 800 --dec-x 600 --enc-z3 200 --dec-z1 200 --train-w-noise --batch-size 150 --rseed 123