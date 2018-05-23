### >>> DrVAE 6h: <<<
python pool_results.py --files 03-12-2018DrVAE-strictC2C-FDAdrugsv2.1-6h-archC-pertloss0.05-pertkl1-lr0005-bs150-wnoise/results/DrVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "X1->Y" "X1-wI->Y" "Z1->Y" "PCA_X1->Y" --model DGM RF100 Ridge SVMrbf --drug all --measure AUPR AUROC Acc --sep csv
python pool_results.py --files 03-12-2018DrVAE-strictC2C-FDAdrugsv2.1-6h-archC-pertloss0.05-pertkl1-lr0005-bs150-wnoise/results/DrVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "REC_X1" "PERT_X2" "PERTwI_X2" "REC_X2" "PERT_Z1_trueZ2" "PERT_Z2_trueZ2" --model DGM PCA --measure AUROC R2 PearR RMSE KL logL --valid

### >> SSVAE: <<<
python pool_results.py --files 03-12-2018SSVAE-strictC2C-FDAdrugsv2.1-6h-archC-alldata-lr0005-bs150-wnoise/results/SSVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "X1->Y" "Z1->Y" "PCA_X1->Y" --model DGM RF100 Ridge SVMrbf
python pool_results.py --files 03-12-2018SSVAE-strictC2C-FDAdrugsv2.1-6h-archC-alldata-lr0005-bs150-wnoise/results/SSVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "REC_X1" --model DGM --measure AUROC R2 PearR --valid

### >> DrVAE 24h: <<<
python pool_results.py --files 03-12-2018DrVAE-strictC2C-FDAdrugsv2.1-24h-archC-pertloss0.05-pertkl1-lr0005-bs150-wnoise/results/DrVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "X1->Y" "X1-wI->Y" "Z1->Y" "PCA_X1->Y" --model DGM RF100 Ridge SVMrbf

### >>> PertVAE: <<<
# --> test classification on top of learned latent space on held out test set
python pool_results.py --files 04-05-2018PVAE-FDAdrugsv2.1-6h-archC-pertloss0.05-lr0005-bs150-woPairTest-wnoise/results/PVAE_SD_test_all_RS*_L2_KLR*_FOLD* --task "X1->Y" "X1-wI->Y" "Z1->Y" "PCA_X1->Y" --model DGM RF100 Ridge SVMrbf --measure AUPR AUROC Acc --drug all --sep csv


### >>> New pooling for Wilcoxon test
python pool_results.py --files 03-12-2018DrVAE-strictC2C-FDAdrugsv2.1-6h-archC-pertloss0.05-pertkl1-lr0005-bs150-wnoise/results/DrVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "X1->Y" "X1-wI->Y" "Z1->Y" "PCA_X1->Y" --model DGM RF100 Ridge SVMrbf --measure AUPR AUROC Acc --drug all --sep csv --dump-rlist DrVAE
python pool_results.py --files 03-12-2018SSVAE-strictC2C-FDAdrugsv2.1-6h-archC-alldata-lr0005-bs150-wnoise/results/SSVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "X1->Y" "Z1->Y" --model DGM RF100 Ridge SVMrbf --measure AUPR AUROC Acc --drug all --sep csv --dump-rlist SSVAE
python pool_results.py --files 04-05-2018PVAE-FDAdrugsv2.1-6h-archC-pertloss0.05-lr0005-bs150-woPairTest-wnoise/results/PVAE_SD_test_all_RS*_L2_KLR*_FOLD* --task "X1->Y" "Z1->Y" "PCA_X1->Y" --model RF100 Ridge SVMrbf --measure AUPR AUROC Acc --drug all --sep csv --dump-rlist PertVAE

python ../src/scripts/compare_pooled_results.py --files 03-12-2018DrVAE-strictC2C-FDAdrugsv2.1-6h-archC-pertloss0.05-pertkl1-lr0005-bs150-wnoise-100-rlist.pkl 03-12-2018SSVAE-strictC2C-FDAdrugsv2.1-6h-archC-alldata-lr0005-bs150-wnoise-100-rlist.pkl 04-05-2018PVAE-FDAdrugsv2.1-6h-archC-pertloss0.05-lr0005-bs150-woPairTest-wnoise-100-rlist.pkl --task "X1->Y" "X1-wI->Y" "Z1->Y" "PCA_X1->Y"  --drug all --measure AUPR AUROC Acc --wilcoxon-allvsall --sep csv > 03-12-2018DrVAE6h-wilcoxon-allvsall-03-12-2018SSVAE-04-05-2018PVAE.csv
python ../src/scripts/compare_pooled_results.py --files 03-12-2018DrVAE-strictC2C-FDAdrugsv2.1-6h-archC-pertloss0.05-pertkl1-lr0005-bs150-wnoise-100-rlist.pkl 03-12-2018SSVAE-strictC2C-FDAdrugsv2.1-6h-archC-alldata-lr0005-bs150-wnoise-100-rlist.pkl 04-05-2018PVAE-FDAdrugsv2.1-6h-archC-pertloss0.05-lr0005-bs150-woPairTest-wnoise-100-rlist.pkl --task "X1->Y" "X1-wI->Y" "Z1->Y" "PCA_X1->Y" --model DrVAE-DGM --drug 26 --measure AUPR AUROC --wilcoxon-allvsall --sep csv > 03-12-2018DrVAE6h-wilcoxon-vs-03-12-2018SSVAE-04-05-2018PVAE.csv


python pool_results.py --files 04-15-2018DrVAE-strictC2C-FDAdrugsv2.1-6h-archC-pertloss0.05-pertkl1-dataprior-wnoise-rnd*/results/DrVAE_SD_*_all_RS*_L2_YR*_FOLD* --task "X1->Y" "X1-wI->Y" "Z1->Y" "PCA_X1->Y" --model DGM RF100 Ridge SVMrbf --measure AUPR AUROC PPV
