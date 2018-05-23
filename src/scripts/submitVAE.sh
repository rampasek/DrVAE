##---------- SSVAE_SD -----------
DATA="datafiles/CTRPv2+L1000_FDAdrugs6h_v2.1.h5"
ODIR="03-12-2018SSVAE-strictC2C-FDAdrugsv2.1-6h-archC-alldata-lr0005-bs150-wnoise"
if [ ! -d $ODIR/logs ]; then mkdir -p $ODIR/logs; fi
for L in 2; do
for suprate in 1 3 5 10; do
for i in {1..5}; do
for RS in 1636 6997 754 5189 2005 1295 6237 5535 9716 2858 5854 9983 2143 9488 850 4451 708 7267 2243 1776; do
modelID='RS'$RS'_L'$L'_YR'$suprate'_FOLD'$i;
COMMONPARAMS="--datafile $DATA --modelid $modelID --rseed $RS --stopearly --L $L --yloss-rate $suprate --fold $i --outdir $ODIR"
PARAMS="${COMMONPARAMS} --dim-z1 100 --dim-z2 100 --enc-z1 800 --dec-x 600 --enc-z2 200 --dec-z1 200 --alldata --train-w-noise --batch-size 150" #archC
# python ~/DrVAE/workspace/run_vfae.py $PARAMS > $ODIR/out_$modelID.txt 2>&1 &
# sbatch -c 4 -D ~/DrVAE/workspace -o $ODIR'/logs/VFAE_C2C_'$modelID'.out' ~/DrVAE/workspace/run_vfae.py $PARAMS;
qsub -V -l nodes=1:ppn=4,vmem=6g,walltime=10:00:00 -d ~/laci/DrVAE/workspace -j oe -o $ODIR'/logs/VFAE_C2C_'$modelID'.out' ~/laci/DrVAE/workspace/run_vfae.py -F "$PARAMS";
done; done; done; done;


##---------- DrVAE_SD
DATA="datafiles/CTRPv2+L1000_FDAdrugs6h_v2.1.h5"
ODIR="03-12-2018DrVAE-strictC2C-FDAdrugsv2.1-6h-archC-pertloss0.05-pertkl1-lr0005-bs150-wnoise"
if [ ! -d $ODIR/logs ]; then mkdir -p $ODIR/logs; fi
for L in 2; do
for suprate in 1 3 5 10; do
for i in {1..5}; do
for RS in 1636 6997 754 5189 2005 1295 6237 5535 9716 2858 5854 9983 2143 9488 850 4451 708 7267 2243 1776; do
modelID='RS'$RS'_L'$L'_YR'$suprate'_FOLD'$i;
COMMONPARAMS="--datafile $DATA --modelid $modelID --rseed $RS --stopearly --L $L --yloss-rate $suprate --fold $i --outdir $ODIR"
PARAMS="${COMMONPARAMS} --dim-z1 100 --dim-z3 100 --enc-z1 800 --dec-x 600 --enc-z3 200 --dec-z1 200 --train-w-noise --batch-size 150 --test-only" #archC
# sbatch -c 4 -D ~/DrVAE/workspace -o $ODIR'/logs/DrVAE_C2C_'$modelID'.out' ~/DrVAE/workspace/run_drvae.py $PARAMS;
qsub -V -l nodes=1:ppn=4,vmem=6g,walltime=10:00:00 -d ~/laci/DrVAE/workspace -j oe -o $ODIR'/logs/DrVAE_C2C_'$modelID'.out' ~/laci/DrVAE/workspace/run_drvae.py -F "$PARAMS";
done; done; done; done;


##---------- Pert VAE
DATA="datafiles/CTRPv2+L1000_FDAdrugs6h_v2.1.h5"
ODIR="04-05-2018PVAE-FDAdrugsv2.1-6h-archC-pertloss0.05-lr0005-bs150-woPairTest-wnoise"
if [ ! -d $ODIR/logs ]; then mkdir -p $ODIR/logs; fi
for L in 2; do
for klrate in 1; do
for i in {1..5}; do
for RS in 1636 6997 754 5189 2005 1295 6237 5535 9716 2858 5854 9983 2143 9488 850 4451 708 7267 2243 1776; do
modelID='RS'$RS'_L'$L'_KLR'$klrate'_FOLD'$i;
COMMONPARAMS="--datafile $DATA --modelid $modelID --rseed $RS --stopearly --L $L --kl-z2-rate $klrate --fold $i --outdir $ODIR"
PARAMS="${COMMONPARAMS} --dim-z1 100 --enc-z1 800 --dec-x 600 --train-w-noise --batch-size 150" #archC   --with-pairdata-test
qsub -V -l nodes=1:ppn=4,vmem=6g,walltime=10:00:00 -d ~/laci/DrVAE/workspace -j oe -o $ODIR'/logs/PVAE_C2C_'$modelID'.out' ~/laci/DrVAE/workspace/run_pvae.py -F "$PARAMS";
done; done; done; done;