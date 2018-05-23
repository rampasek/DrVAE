#!/usr/bin/env python
"""
Ladislav Rampasek (rampasek@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
## add src path that is missing when running with Slurm 
sys.path.append(os.path.dirname(os.getcwd())+"/src")
## print python version, machine hostname, and hash+summary of the latest git commit
print('hostname:', os.uname()[1])
print(sys.version)
os.system('git log -1 |cat')

import argparse
import numpy as np
import sklearn.utils
import sklearn.metrics
from scipy import stats
from collections import Counter, OrderedDict
from pprint import pprint
import json

import torch
import torch.utils.data
from torch.autograd import Variable

from PVAE import PVAE
import utils as utl
# data set format is shared with DrVAE
from DrVAE import DrVAEDataset, wrap_in_DrVAEDataset

# enforce pytorch version 0.3.x, refactoring is required for 0.4.x
print('pytorch version:', torch.__version__)
if not torch.__version__.startswith('0.3'):
    raise Exception('pytorch version 0.3.x is required')
# set number of CPU parallel threads to 4, performance doesn't scale beyond 4
print('orig num threads:', torch.get_num_threads())
torch.set_num_threads(4)
print('now num threads:', torch.get_num_threads())
print('-----')


def main(args):
    np.set_printoptions(precision=3, suppress=True)

    ## LOAD data
    if args.datafile.endswith('.RData'):
        data = utl.load_from_RData(args.datafile)
        # datafile_hdf = args.datafile.replace('_rpy2.RData', '.h5')
        # utl.save_to_HDF(datafile_hdf, data)
        # print('Saved data to:', datafile_hdf)
        # return
    else:
        data = utl.load_from_HDF(args.datafile)
    print('Loaded data from:', args.datafile)

    ## create output directories
    utl.make_out_dirs(args.outdir)

    ## drug selection
    drug_list_26 = ['omacetaxine mepesuccinate', 'bortezomib', 'vorinostat', 'paclitaxel', 'docetaxel', 'topotecan',
                    'niclosamide', 'valdecoxib','teniposide', 'vincristine', 'prochlorperazine', 'mitomycin', 'lovastatin',
                    'gemcitabine', 'dasatinib', 'fluvastatin', 'clofarabine', 'sirolimus', 'etoposide', 'sitagliptin',
                    'decitabine', 'PLX-4032', 'fulvestrant', 'bosutinib', 'trifluoperazine', 'ciclosporin']
    drug_list_26 = sorted(drug_list_26)
    if args.drug == 'all':
        drug_list = drug_list_26
        for d in sorted(data['drug_drug']):
            if d not in drug_list:
                drug_list.append(d)
    elif args.drug == '26':
        drug_list = drug_list_26
    else:
        if args.drug in data['drug_drug']:
            drug_list = [args.drug]
        else:
            raise ValueError('Selected drug not found: ' + args.drug)

    all_stats = {'train': dict(), 'valid': dict(), 'test': dict()}
    for selected_drug in drug_list:
        ## ignore drugs that don't have enough perturbations
        if selected_drug in ["abiraterone", "azacitidine", "cyclophosphamide", "methotrexate", "fluorouracil",
                             "ifosfamide", "ciclopirox"]:
            print("Ignoring: ", selected_drug)
            continue
        ## initialize random state
        rnds = sklearn.utils.check_random_state(args.rseed)
        np.random.seed(args.rseed)
        torch.manual_seed(args.rseed)
        if args.cuda:
            torch.cuda.manual_seed(args.rseed)

        ### select the drug data and get CV-split of all data types
        y_unlab_token = -47
        y_key = 'y' if args.type_y == 'discrete' else 'ycont'
        sing, singv, singt, pair, pairv, pairt = utl.split_data_sd(
            data, selected_drug, args.data_mode, fold=args.fold, n_folds=5, rnds=rnds,
            noPairTest=not args.with_pairdata_test, unlab_token=y_unlab_token, verbose=False
        )
        concat_flag = 'pair_only' if args.pair_data_only else 'both'
        train_dataset, train_ddict = wrap_in_DrVAEDataset(sing, pair, y_key, concat=concat_flag,
                                                          downlabel_to=args.downlabel_to,
                                                          remove_unlabeled=False)
        valid_dataset, valid_ddict = wrap_in_DrVAEDataset(singv, pairv, y_key, concat=concat_flag,
                                                          remove_unlabeled=False)
        test_dataset, test_ddict = wrap_in_DrVAEDataset(singt, pairt, y_key, concat=concat_flag)

        N = len(train_dataset)
        dim_x = sing['x1'].shape[1]
        dim_s = np.unique(sing['s']).shape[0]
        if args.type_y == 'discrete':
            class_sizes = np.bincount(sing['y'][ sing['has_y'] ])
            print(class_sizes)
            dim_y = len(class_sizes)
            data_prior_y = class_sizes / (1.*sum(class_sizes))
        else:
            dim_y = 1
            _tmp_ycont = sing['ycont'][ sing['has_y'] ]
            data_prior_y = np.array([_tmp_ycont.mean(), _tmp_ycont.std()])
        print('train data prior: ', data_prior_y)

        if args.modelid in ["auto", "'auto'"]:
            args.modelid = 'RS{}_L{}_KLZ2{:.0f}_FOLD{}'.format(args.rseed, args.L, args.kl_z2_rate, args.fold)

        print("PertVAE on a single drug, modelid: ", args.modelid)
        print("selected_drug: ", selected_drug)
        print("keep paired data test fold: ", args.with_pairdata_test)
        print("concat_flag: ", concat_flag)
        print(N, dim_x, dim_y)
        print("sensitivity data")
        if args.type_y == 'discrete':
            print("    train data (Y, S):", Counter(sing['y']), Counter(sing['s']))
            print("    valid data (Y, S):", Counter(singv['y']), Counter(singv['s']))
            print("    test data  (Y, S):", Counter(singt['y']), Counter(singt['s']))
        else:
            print("    train data (Ycont):", stats.describe(sing['ycont'][sing['ycont'] != y_unlab_token]))
            print("    valid data (Ycont):", stats.describe(singv['ycont'][singv['ycont'] != y_unlab_token]))
            print("    test data  (Ycont):", stats.describe(singt['ycont'][singt['ycont'] != y_unlab_token]))

        ### create balanced sampler for training set
        ## balance by cell line ids
        balanced_weights = utl.compute_balanced_weights(utl.cid2numid(train_ddict['cid']), unlabeled_data_ratio=None)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(balanced_weights, len(balanced_weights))

        ### create balanced sampler for validation set
        ## balance validation sampler by cell line ids
        balanced_weights = utl.compute_balanced_weights(utl.cid2numid(valid_ddict['cid']), unlabeled_data_ratio=None)
        valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(balanced_weights, len(balanced_weights))
        
        ## create train/valid/test data loaders
        dl_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        dl_kwargs['batch_size'] = args.batch_size
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   drop_last=(len(train_dataset) >= args.batch_size),
                                                   sampler=train_sampler, **dl_kwargs)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   drop_last=(len(valid_dataset) >= args.batch_size),
                                                   sampler=valid_sampler, **dl_kwargs)
        
        print('Train set length: {} batches; {} examples'.format(len(train_loader), len(train_loader.dataset)))
        print('Valid set length: {} batches; {} examples'.format(len(valid_loader), len(valid_loader.dataset)))
        print('Test set length: {} examples'.format(len(test_dataset)))
            
        model_filename = 'models/PVAE_SD_{}_{}.pth'.format(args.modelid, selected_drug)
        model = PVAE(dim_x=dim_x, dim_s=dim_s, dim_y=dim_y, dim_z1=args.dim_z1,
                     dim_h_en_z1=args.enc_z1, dim_h_en_z2Fz1=args.enc_z2Fz1,
                     dim_h_de_x=args.dec_x, type_rec='diag_gaussian',
                     optim_alg='adam', batch_size=args.batch_size, epochs=1000,
                     nonlinearity='elu', L=args.L,
                     weight_decay=0.05, dropout_rate=0., input_x_dropout=args.x_dropout, add_noise_var=args.noise_var,
                     learning_rate=0.0005,
                     kl_qz2pz2_rate=args.kl_z2_rate, pertloss_rate=0.05, anneal_perturb_rate_itermax=1, anneal_perturb_rate_offset=0,
                     use_s=args.useS, use_MMD=args.useMMD, kernel_MMD='rbf_fourier', mmd_rate=args.mmd_rate,  ## <= settings for "fairness" (like VFAE)
                     random_seed=args.rseed, log_txt=None  #'PVAE_SD_{}_{}.txt'.format(args.modelid, selected_drug)
                     )
        if args.cuda:
            model.cuda()

        if not args.test_only:
            model.fit(train_loader=train_loader, valid_loader=valid_loader, add_noise=args.train_w_noise,
                      verbose=True, early_stop=args.stopearly, model_filename=model_filename)
    
        print ('Loading trained model from file...')
        model.load_params_from_file(model_filename)
        print ('...done.')

        #### test the model and baselines
        print ('Testing...')
        ## run our model
        model_train_perf, train_perfstr = model.evaluate_performance_on_dataset(train_dataset, return_full_data=True)
        model_valid_perf, valid_perfstr = model.evaluate_performance_on_dataset(valid_dataset, return_full_data=True)
        model_test_perf, test_perfstr = model.evaluate_performance_on_dataset(test_dataset, return_full_data=True)
        model.w2log('Train set performance:\t', train_perfstr)
        model.w2log('Valid set performance:\t', valid_perfstr)
        model.w2log('Test set performance:\t', test_perfstr)

        ## run baselines
        # all_stats['train'][selected_drug] = utl.compile_baseline_stats(args.type_y,
        #                                                         tr=train_ddict, ev=train_ddict,
        #                                                         model_tr=model_train_perf, model_ev=model_train_perf,
        #                                                         svmkernel='rbf', rseed=args.rseed)
        all_stats['valid'][selected_drug] = utl.compile_baseline_stats(args.type_y,
                                                                tr=train_ddict, ev=valid_ddict,
                                                                model_tr=model_train_perf, model_ev=model_valid_perf,
                                                                svmkernel='rbf', rseed=args.rseed)
        all_stats['test'][selected_drug] = utl.compile_baseline_stats(args.type_y,
                                                                tr=train_ddict, ev=test_ddict,
                                                                model_tr=model_train_perf, model_ev=model_test_perf,
                                                                svmkernel='rbf', rseed=args.rseed)
        ## print the stats
        for k in all_stats['test'][selected_drug].keys():
            valid_v = all_stats['valid'][selected_drug][k]
            test_v = all_stats['test'][selected_drug][k]
            k = k.replace('|', '\t')
            if isinstance(valid_v, float):
                print('{}\t{:.4f}\t{:.4f}'.format(k, valid_v, test_v))
            else:
                print(k, valid_v, test_v)

        ## save performance reports as json
        for evalset in ('valid', 'test'):
            ## all stats
            results_fname = 'results/PVAE_SD_{}_all_{}.json'.format(evalset, args.modelid)
            with open(results_fname, 'wb') as f:
                jd = json.dumps(all_stats[evalset], sort_keys=False, indent=4, separators=(',', ': '))
                f.write(jd.encode())

    return all_stats


if __name__ == '__main__':
    ### Parse command line arguments
    parser = argparse.ArgumentParser(description='Drug perturbation VAE (PertVAE)')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--modelid', type=str, required=True, help='model ID')
    parser.add_argument('--datafile', type=str, required=True, help='input data file')
    parser.add_argument('--outdir', type=str, default=None, help='output directory')
    parser.add_argument('--fold', type=int, default=1, help='which data fold to run (1 to 5)')
    parser.add_argument('--test-only', action='store_true', default=False, help='load saved parameters and run tests')
    parser.add_argument('--batch-size', type=int, default=200, help='minibatch size')
    parser.add_argument('--L', type=int, default=1, help='number of samples from Q (default 1)')
    parser.add_argument('--stopearly', action="store_true", dest="stopearly", default=False, help='train with early stopping (default False)')
    parser.add_argument('--kl-z2-rate', type=float, default=1., help='weight of KL loss of matching q(z2|x2) to p(z2|z1)')
    parser.add_argument('--rseed', type=int, default=12345, help='random seed')
    parser.add_argument('--use-s', action="store_true", dest="useS", default=False, help='use model with nuisance variable S')
    parser.add_argument('--use-mmd', action="store_true", dest="useMMD", default=False, help='include MMD loss to improve independence of Z from S')
    parser.add_argument('--no-mf', action="store_true", dest="useMF", default=False, help='use molecular features of drugs')
    parser.add_argument('--mmd-rate', type=float, default=1., help='weight of MMD loss')
    parser.add_argument('--data-mode', type=str, default="strictC2C", help='which data to use for training and testing')
    parser.add_argument('--drug', type=str, default="26", help='select one drug to run or set of "26" or "all"')
    parser.add_argument('--pair-data-only', action='store_true', default=False, help='use only perturbation pair data (ignore singletons)')
    parser.add_argument('--train-w-noise', action='store_true', default=False, help='add random Gaussian noise to the gene expression during training')
    parser.add_argument('--with-pairdata-test', action='store_true', default=False, help='keep a hold out fold of perturbation pairs for testing')
    parser.add_argument('--noise-var', type=float, default=0.01, help='variance of the Gaussian noise to agument input data')
    parser.add_argument('--x-dropout', type=float, default=0., help='Input dropout rate')
    parser.add_argument('--downlabel-to', type=int, default=None, help='Reduce number of labeled data in training set to given number by masking labeled data as unlabeled')
    # architecture:
    parser.add_argument('--dim-z1', type=int, default=50, help='size of z1 & z2')
    parser.add_argument('--enc-z1', type=int, nargs='+', default=[200,200], help='NN size of encoder q(z_k|x_k)')
    parser.add_argument('--enc-z2Fz1', type=int, nargs='+', default=[], help='NN size of encoder p(z2|z1)')
    parser.add_argument('--dec-x', type=int, nargs='+', default=[200,200], help='NN size of decoder p(x_k|z_k)')
    parser.add_argument('--type-y', type=str, default='discrete', help='("discrete" or "cont"): classification vs regression')

    args = parser.parse_args()
    print(args)

    # args.cuda = args.cuda and torch.cuda.is_available()    
    assert (args.cuda == False), 'GPU is not supported yet'

    availableDataModes = ['strictC2C'] ## no patient data used; even in unlabeled data
    assert (args.data_mode in availableDataModes), 'Unsupported data mode'

    assert (args.type_y in ['discrete', 'cont']), 'Invalid type-y'
    assert (args.useMF == False), 'use of mol.features not supported'

    main(args)

