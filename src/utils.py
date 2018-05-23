"""
Ladislav Rampasek (rampasek@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import h5py
import json
import numpy as np
import sklearn.utils
import sklearn.metrics
import sklearn.preprocessing
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats

from collections import Counter, OrderedDict
import torch


def random_chance(y):
    return max(np.bincount(y.astype(int)) / float(y.shape[0]))


def concat_dicts(a, b):
    # works for both dict and OrderedDict classes
    assert (len(set(a.keys()) & set(b.keys())) == 0), "Can't unambiguously concatenate."
    return type(a)(list(a.items()) + list(b.items()))


def reduce_dimensionality(train, test, n_components, method='PCA', rseed=None):
    if method == 'PCA':
        from sklearn.decomposition import PCA
        model = PCA(n_components=n_components, random_state=rseed)
    elif method == 'TSNE':
        from sklearn.manifold import TSNE
        model = TSNE(n_components=n_components, random_state=rseed)
    else:
        raise ValueError("Unknown method: " + str(method))
    # transformed = model.fit_transform( np.vstack((train, test)) )
    train_transformed = model.fit_transform(train)
    test_transformed = model.transform(test)
    return train_transformed, test_transformed


def eval_PCA_reconstruction(train, test, n_components, rseed=None):
    from sklearn.decomposition import PCA
    model = PCA(n_components=n_components, random_state=rseed)
    model.fit(train)
    test_transformed = model.inverse_transform(model.transform(test))

    perf = OrderedDict()
    perf['RMSE|PCA'] = np.sqrt(((test - test_transformed) ** 2).mean())
    perf['R2|PCA'] = sklearn.metrics.r2_score(test, test_transformed)
    num_examples = float(test.shape[0])
    perf['PearR|PCA'] = 0.
    for i in range(test.shape[0]):
        perf['PearR|PCA'] += stats.pearsonr(test[i, :], test_transformed[i, :])[0] / num_examples
    return perf


def compute_cl_metrics(name, preds, proba, ytest):
    perf = OrderedDict()
    perf['Acc|' + name] = (preds == ytest).mean()
    # perf['F1|' + name] = sklearn.metrics.f1_score(ytest, preds)
    # perf['PPV|' + name] = (ytest[preds == 1] == 1).mean()
    try:
        perf['AUROC|' + name] = sklearn.metrics.roc_auc_score(ytest, proba)
    except:
        perf['AUROC|' + name] = np.nan
    try:
        perf['AUPR|' + name] = sklearn.metrics.average_precision_score(ytest, proba)
    except:
        perf['AUPR|' + name] = np.nan
    return perf


def measure_cl_perf(cl, name, xtrain, ytrain, xtest, ytest):
    # fit sklearn classifier
    cl.fit(xtrain, ytrain)
    preds = cl.predict(xtest)
    proba = cl.predict_proba(xtest)[:, 1]
    # evaluate
    return compute_cl_metrics(name, preds, proba, ytest)


def classif_baseline_perf(xtrain, ytrain, xtest, ytest, svmkernel='rbf', rseed=None):
    data = [xtrain, ytrain, xtest, ytest]
    perf = OrderedDict()

    ## Random Forest
    cl = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=rseed)
    rf_perf = measure_cl_perf(cl, 'RF100', *data)
    perf = concat_dicts(perf, rf_perf)

    # cl = sklearn.ensemble.RandomForestClassifier(n_estimators=50, random_state=rseed)
    # rf_perf = measure_cl_perf(cl, 'RF50', *data)
    # perf = concat_dicts(perf, rf_perf)

    cl = sklearn.linear_model.LogisticRegressionCV(penalty='l2', random_state=rseed)
    lr_perf = measure_cl_perf(cl, 'Ridge', *data)
    perf = concat_dicts(perf, lr_perf)

    ## SVM 
    cl = sklearn.svm.SVC(kernel=svmkernel, probability=True, random_state=rseed)
    svm_perf = measure_cl_perf(cl, 'SVM' + svmkernel[:3], *data)
    perf = concat_dicts(perf, svm_perf)
    # SVM w/ linear kernel
    cl = sklearn.svm.SVC(kernel='linear', probability=True, random_state=rseed)
    svm_perf = measure_cl_perf(cl, 'SVMlin', *data)
    perf = concat_dicts(perf, svm_perf)

    # ## Random
    # rnd_proba = np.random.rand(*ytest.shape)
    # rnd_preds = (rnd_proba > 0.5).astype(int)
    # rnd_perf = compute_cl_metrics('rnd', rnd_preds, rnd_proba, ytest)
    # perf = concat_dicts(perf, rnd_perf)
    return perf


def measure_reg_perf(reg, name, xtrain, ytrain, xtest, ytest):
    # fit sklearn classifier
    reg.fit(xtrain, ytrain)
    preds = reg.predict(xtest)
    # evaluate
    perf = OrderedDict()
    perf['RMSE|' + name] = np.sqrt(((ytest - preds) ** 2).mean())
    perf['R2|' + name] = sklearn.metrics.r2_score(ytest, preds)
    perf['PearR|' + name] = stats.pearsonr(ytest, preds)[0]
    return perf


def regression_baseline_perf(xtrain, ytrain, xtest, ytest, svmkernel='rbf', rseed=None):
    data = [xtrain, ytrain, xtest, ytest]
    perf = OrderedDict()

    ## Random Forest
    reg = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=rseed)
    rf_perf = measure_reg_perf(reg, 'RF100', *data)
    perf = concat_dicts(perf, rf_perf)


    ## L2 Linear Regression
    reg = sklearn.linear_model.RidgeCV()
    lr_perf = measure_reg_perf(reg, 'Ridge', *data)
    perf = concat_dicts(perf, lr_perf)

    ## SVM 
    reg = sklearn.svm.SVR(kernel=svmkernel)
    svm_perf = measure_reg_perf(reg, 'SVM' + svmkernel[:3], *data)
    perf = concat_dicts(perf, svm_perf)
    # SVM w/ linear kernel
    reg = sklearn.svm.SVR(kernel='linear')
    svm_perf = measure_reg_perf(reg, 'SVMlin', *data)
    perf = concat_dicts(perf, svm_perf)

    return perf


def run_baseline_models(type_y, xtrain, ytrain, xtest, ytest, svmkernel='rbf', rseed=None):
    if type_y == 'discrete':
        perf = classif_baseline_perf(xtrain, ytrain, xtest, ytest, svmkernel, rseed)
    else:
        perf = regression_baseline_perf(xtrain, ytrain, xtest, ytest, svmkernel, rseed)
    return perf


def compile_baseline_stats(type_y, tr, ev, model_tr, model_ev, svmkernel='rbf', rseed=None):
    s = OrderedDict()
    # s['Train|Random Chance|S'] = random_chance(tr['s'])
    # s['Eval|Random Chance|S'] = random_chance(ev['s'])
    # uniq_tids = np.unique(np.concatenate((tr['tid'], ev['tid'])))
    # s['Train|Counts|TissueIds'] = [(tr['tid'] == _t).sum() for _t in uniq_tids]
    # s['Eval|Counts|TissueIds'] = [(ev['tid'] == _t).sum() for _t in uniq_tids]

    if type_y == 'discrete':
        tr_ylab = tr['y'][tr['has_y']]
        ev_ylab = ev['y'][ev['has_y']]
        s['Train|Random Chance|Y'] = random_chance(tr_ylab)
        s['Eval|Random Chance|Y'] = random_chance(ev_ylab)
    else:
        tr_ylab = tr['ycont'][tr['has_y']]
        ev_ylab = ev['ycont'][ev['has_y']]
        s['Train|Random Chance|Y'] = tr_ylab.mean()
        s['Eval|Random Chance|Y'] = ev_ylab.mean()

    ## Run baselines on the original X1 -> Y
    x1_stats = run_baseline_models(type_y,
                                   tr['x1'][tr['has_y']], tr_ylab,
                                   ev['x1'][ev['has_y']], ev_ylab,
                                   rseed=rseed, svmkernel=svmkernel)
    x1_stats = OrderedDict([('X1->Y|' + k, v) for k, v in x1_stats.items()])
    s = concat_dicts(s, x1_stats)

    ## Run baselines on PCA embedding of X1 -> Y
    # use both labeled and unlabeled training data to fit
    dim_z1 = model_tr['z1'].shape[1]
    print('Using {} PCs'.format(dim_z1))
    tr_x1_pca, ev_x1_pca = reduce_dimensionality(tr['x1'], ev['x1'], n_components=dim_z1, method='PCA')
    x1pca_stats = run_baseline_models(type_y,
                                      tr_x1_pca[tr['has_y']], tr_ylab,
                                      ev_x1_pca[ev['has_y']], ev_ylab,
                                      rseed=rseed, svmkernel=svmkernel)
    x1pca_stats = OrderedDict([('PCA_X1->Y|' + k, v) for k, v in x1pca_stats.items()])
    s = concat_dicts(s, x1pca_stats)

    ## Run baselines on Z1 representation of X1; Z1 -> Y
    assert model_tr['z1'].shape[0] == tr['x1'].shape[0]
    assert model_ev['z1'].shape[0] == ev['x1'].shape[0]
    z1_stats = run_baseline_models(type_y,
                                   model_tr['z1'][tr['has_y']], tr_ylab,
                                   model_ev['z1'][ev['has_y']], ev_ylab,
                                   rseed=rseed, svmkernel=svmkernel)
    z1_stats = OrderedDict([('Z1->Y|' + k, v) for k, v in z1_stats.items()])
    s = concat_dicts(s, z1_stats)
    
    if model_ev['model_class'] in ['DrVAE', 'PVAE']:
        ## Run baselines on Z2 representation of X1; Z2 -> Y
        assert model_tr['z2'].shape[0] == tr['x1'].shape[0]
        assert model_ev['z2'].shape[0] == ev['x1'].shape[0]
        z2_stats = run_baseline_models(type_y,
                                       model_tr['z2'][tr['has_y']], tr_ylab,
                                       model_ev['z2'][ev['has_y']], ev_ylab,
                                       rseed=rseed, svmkernel=svmkernel)
        z2_stats = OrderedDict([('Z2->Y|' + k, v) for k, v in z2_stats.items()])
        s = concat_dicts(s, z2_stats)
        ## Run baselines on Z1,Z2-Z1 representation of X1; Z1Z2 -> Y
        z1z2_stats = run_baseline_models(type_y,
            np.hstack((model_tr['z1'][tr['has_y']], model_tr['z2'][tr['has_y']] - model_tr['z1'][tr['has_y']])), tr_ylab,
            np.hstack((model_ev['z1'][ev['has_y']], model_ev['z2'][ev['has_y']] - model_ev['z1'][ev['has_y']])), ev_ylab,
            rseed=rseed, svmkernel=svmkernel)
        z1z2_stats = OrderedDict([('Z1Z2->Y|' + k, v) for k, v in z1z2_stats.items()])
        s = concat_dicts(s, z1z2_stats)

    s['model_class'] = model_ev['model_class']
    s['type_y'] = type_y
    ## Include supplied @model_ev performance X1 -> Y
    if model_ev['model_class'] in ['VFAE', 'DrVAE']:
        if type_y == 'discrete':
            s['X1->Y|Acc|DGM'] = model_ev['y_acc']
            s['X1->Y|AUROC|DGM'] = model_ev['y_auroc']
            s['X1->Y|AUPR|DGM'] = model_ev['y_aupr']
        else:
            s['X1->Y|RMSE|DGM'] = model_ev['y_rmse']
            s['X1->Y|R2|DGM'] = model_ev['y_r2']
            s['X1->Y|PearR|DGM'] = model_ev['y_pearr']
    # results after setting perturbation function equal to identity
    if model_ev['model_class'] in ['DrVAE']: 
        if type_y == 'discrete':
            s['X1-wI->Y|Acc|DGM'] = model_ev['y_wI_acc']
            s['X1-wI->Y|AUROC|DGM'] = model_ev['y_wI_auroc']
            s['X1-wI->Y|AUPR|DGM'] = model_ev['y_wI_aupr']

    ## Include supplied @model_ev X1 reconstruction performance
    s['REC_X1|RMSE|DGM'] = model_ev['x1_rmse']
    s['REC_X1|R2|DGM'] = model_ev['x1_r2']
    s['REC_X1|PearR|DGM'] = model_ev['x1_pearr']
    s['REC_X1|logL|DGM'] = model_ev['x1_ll']
    pcax1_stats = eval_PCA_reconstruction(tr['x1'], ev['x1'], n_components=dim_z1, rseed=rseed)
    pcax1_stats = OrderedDict([('REC_X1|' + k, v) for k, v in pcax1_stats.items()])
    s = concat_dicts(s, pcax1_stats)

    ## Include supplied @model_ev X2 prediction performance
    if model_ev['model_class'] in ['DrVAE', 'PVAE']:
        s['PERT_X2|RMSE|DGM'] = model_ev['x2_rmse']
        s['PERT_X2|R2|DGM'] = model_ev['x2_r2']
        s['PERT_X2|PearR|DGM'] = model_ev['x2_pearr']
        s['PERT_X2|logL|DGM'] = model_ev['x2_ll']

        s['REC_X2|RMSE|DGM'] = model_ev['x2_rec_rmse']
        s['REC_X2|R2|DGM'] = model_ev['x2_rec_r2']
        s['REC_X2|PearR|DGM'] = model_ev['x2_rec_pearr']
        s['REC_X2|logL|DGM'] = model_ev['x2_rec_ll']

        # results after setting perturbation function equal to identity
        s['PERTwI_X2|RMSE|DGM'] = model_ev['x2_wI_rmse']
        s['PERTwI_X2|R2|DGM'] = model_ev['x2_wI_r2']
        s['PERTwI_X2|PearR|DGM'] = model_ev['x2_wI_pearr']
        s['PERTwI_X2|logL|DGM'] = model_ev['x2_wI_ll']

        # comparisons of latent embeddings
        s['PERT_Z1_trueZ2|RMSE|DGM'] = model_ev['qz1mu_qz2mu_rmse']
        s['PERT_Z2_trueZ2|RMSE|DGM'] = model_ev['pz2Fz1mu_qz2mu_rmse']
        s['PERT_Z1_trueZ2|KL|DGM'] = model_ev['KL_qz2_qz1']
        s['PERT_Z2_trueZ2|KL|DGM'] = model_ev['KL_qz2_pz2Fz1']
    return s


def compute_balanced_weights(labels, unlabeled_data_ratio=None, unlabeled_token=None):
    """
    Compute sample weights for dataset with @labels such that mini batches
    are expected to have the same number of samples from each class
    @labels: class labels of each sample
    @unlabeled_data_ratio: specify how are unlabeled data sampled if @ds contains any
        None: unlabeled data are sampled with a ratio as if they were another class
        float: unlabeled data are sampled with the given ratio vs labeled data,
               while labeled data are sampled with balanced class ratios
    @unlabeled_token: specifies which class label denotes unlabeled data
    """
    ## if @unlabeled_data_ratio is set, it has to be in (0,1)
    ## and also @unlabeled_token has to be set
    if unlabeled_data_ratio is not None:
        assert unlabeled_token is not None
        assert unlabeled_data_ratio > 0. and unlabeled_data_ratio < 1.

    ## compute number of samples per each class
    class_list = np.unique(labels)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in class_list])
    # print(zip(class_list, class_sample_count))

    ## set weight of each class to be inversely proportional to it's size
    class_weight = 1. / class_sample_count
    ## if specified set labeled vs unlabeled data ratio
    if unlabeled_data_ratio is not None:
        class_weight[class_list != unlabeled_token] *= (1. - unlabeled_data_ratio) / (len(class_list) - 1)
        class_weight[class_list == unlabeled_token] *= unlabeled_data_ratio

    ## get weights for each sample (they don't need to be normalized)
    samples_weight = np.array([class_weight[np.where(class_list == t)[0][0]] for t in labels])
    # print(class_list, class_sample_count, class_weight)

    ## return balanced weights to be used for WeightedRandomSampler
    samples_weight = torch.from_numpy(samples_weight).double()
    return samples_weight


def mkdir_p(path):
    import errno
    import os
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def make_out_dirs(outdir=None):
    import os
    import datetime
    if outdir is not None:
        mkdir_p(outdir)
        os.chdir(outdir)
    else:
        dt = datetime.date.today()
        start_date = "%02d-%02d-%d" % (dt.month, dt.day, dt.year)
        mkdir_p(start_date)
        os.chdir(start_date)
    mkdir_p("models")
    mkdir_p("logs")
    mkdir_p("results")


def selectFromDict(d, keys):
    '''select the chosen @keys in dict @d'''
    res = dict()
    for k in keys:
        if k not in d.keys():
            raise ValueError("Key " + str(k) + " not found")
        else:
            res[k] = d[k]
    return res


def subsetDict(d, ind):
    '''subset each numpy array in dict @d to indices @ind'''
    res = dict()
    if not isinstance(ind, np.ndarray):
        ind = np.asarray(ind)
    for k in d.keys():
        if isinstance(d[k], np.ndarray):
            res[k] = d[k][ind]
    return res


def exportFromDict(d, keys):
    '''return list of items from dict @d'''
    res = list()
    for k in keys:
        if k not in d.keys():
            raise ValueError("Key " + str(k) + " not found")
        else:
            res.append(d[k])
    return res


def concatDictElemetwise(a, b):
    '''concatenate dict @a and @b elementwise'''
    res = dict()
    if not sorted(a.keys()) == sorted(b.keys()):
        raise ValueError("Mismatching key sets")
    for k in a.keys():
        res[k] = np.concatenate((a[k], b[k]), axis=0)
    return res


def cid2numid(string_cids):
    '''
    Transform string keys to numerical id ranging from 0 to "the number of unique keys in @string_cids"
    NOTE: this could be reimplemented by using sklearn.preprocessing.LabelEncoder
    '''
    cid2num = dict()
    for cid in string_cids:
        if cid not in cid2num:
            cid2num[cid] = len(cid2num)
    cid_numerical = np.asarray([cid2num[cid] for cid in string_cids], dtype=int)
    return cid_numerical


def checkSplit(tr, va, te):
    return len(np.intersect1d(tr, te)) == 0 and len(np.intersect1d(tr, va)) == 0 and len(np.intersect1d(te, va)) == 0


def getSplitsByStratifiedKFold(y, fold, n_splits, shuffle, random_state):
    '''stratified splits by @y, i.e. preserving the percentage of samples for each class in @y in each fold'''
    assert (n_splits >= 3)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    kfsplit = kf.split(X=np.zeros(y.shape[0]), y=y)
    allsplits = np.array([x for x in kfsplit])

    if isinstance(fold, list) or isinstance(fold, tuple):
        assert (len(fold) == 2)
        te_fold = fold[0]
        va_fold = fold[1]
    else:
        te_fold = fold
        va_fold = 1
    assert (te_fold >= 1 and te_fold <= n_splits)      ## the test fold is between 1..n_splits
    assert (va_fold >= 1 and va_fold <= n_splits - 1)  ## the validation fold is between 1..(n_splits-1)
    # the te_fold is indexed from 1 so we need to subtract 1
    te_fold_id = te_fold - 1
    va_fold_id = (te_fold_id + va_fold) % n_splits
    ind_te = allsplits[te_fold_id][1]
    ind_va = allsplits[va_fold_id][1]
    ind_tr = np.concatenate(allsplits[np.setdiff1d(np.arange(n_splits), np.asarray([te_fold_id, va_fold_id]))][:, 1])
    assert (checkSplit(ind_tr, ind_va, ind_te))
    return ind_tr, ind_va, ind_te


def getSplitsByGroupKFold(groups, fold, n_splits, shuffle, random_state):
    '''the same group will not appear in two different folds'''
    assert (n_splits >= 3)
    kf = GroupKFold(n_splits=n_splits)

    if shuffle:
        # randomly rename groups so that the GroupKFold (which sorts by group ids first) splits can be randomized
        unique_groups = np.unique(groups)
        rnd_renames = sklearn.utils.shuffle(np.arange(len(unique_groups)), random_state=random_state)
        groups_renamed = np.array([rnd_renames[np.argwhere(unique_groups == g)[0]] for g in groups])
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups_renamed)
    else:
        kfsplit = kf.split(X=np.zeros(groups.shape[0]), groups=groups)
    allsplits = np.array([x for x in kfsplit])

    if isinstance(fold, list) or isinstance(fold, tuple):
        assert (len(fold) == 2)
        te_fold = fold[0]
        va_fold = fold[1]
    else:
        te_fold = fold
        va_fold = 1
    assert (te_fold >= 1 and te_fold <= n_splits)      ## the test fold is between 1..n_splits
    assert (va_fold >= 1 and va_fold <= n_splits - 1)  ## the validation fold is between 1..(n_splits-1)
    # the te_fold is indexed from 1 so we need to subtract 1
    te_fold_id = te_fold - 1
    va_fold_id = (te_fold_id + va_fold) % n_splits
    ind_te = allsplits[te_fold_id][1]
    ind_va = allsplits[va_fold_id][1]
    ind_tr = np.concatenate(allsplits[np.setdiff1d(np.arange(n_splits), np.asarray([te_fold_id, va_fold_id]))][:, 1])
    assert (checkSplit(ind_tr, ind_va, ind_te))
    return ind_tr, ind_va, ind_te


def getSplitsByKFold(y, fold, n_splits, shuffle, random_state):
    '''vanilla K-Folds'''
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    kfsplit = kf.split(X=np.zeros(y.shape[0]), y=y)
    allsplits = np.array([x for x in kfsplit])
    # the @fold is indexed from 1 so we need to subtract 1
    assert (fold >= 1 and fold <= n_splits)
    ind_te = allsplits[fold - 1][1]
    ind_va = allsplits[fold % n_splits][1]
    ind_tr = np.concatenate(allsplits[np.setdiff1d(np.arange(n_splits), np.asarray([fold - 1, fold % n_splits]))][:, 1])
    assert (checkSplit(ind_tr, ind_va, ind_te))
    return ind_tr, ind_va, ind_te


# if __name__ == '__main__':
#     X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]])
#     y = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
#     groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
#     for test_fold in range(1,6):
#         for valid_fold in range(1,5):
#             fold = (test_fold, valid_fold)
#             # ind_tr, ind_va, ind_te = getSplitsByGroupKFold(groups, fold, 5, False, 123)
#             ind_tr, ind_va, ind_te = getSplitsByStratifiedKFold(y, fold, 5, False, 123)
#             print("-----------------", fold)
#             print("train", ind_tr)
#             print("valid", ind_va)
#             print("test", ind_te)

def save_to_HDF(fname, data):
    """Save data (a dictionary) to a HDF5 file."""
    with h5py.File(fname, 'w') as f:
        for key, item in data.items():
            f[key] = item


def load_from_HDF(fname):
    """Load data from a HDF5 file to a dictionary."""
    data = dict()
    with h5py.File(fname, 'r') as f:
        for key in f:
            data[key] = np.asarray(f[key])
            if isinstance(data[key][0], np.bytes_):
                data[key] = data[key].astype(np.str)
            # print(key + ":", f[key])
    return data


def load_from_RData(fname):
    '''
    Load RData file with whole dataset
    expected content of the file:
        sing_x1, sing_cid, sing_tid, sing_s, sing_y, sing_ycont,
        pair_x1, pair_x2, pair_cid, pair_tid, pair_s, pair_y, pair_ycont, pair_drug, pair_m, pair_conc, pair_dur,
        drug_drug, drug_m, drug_threshold,
        labeled_pert_cellid
    '''
    import readline
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri

    robjects.r['load'](fname)
    keys = robjects.r['ls'](robjects.globalenv)
    expected_keys = ["sing_x1", "sing_cid", "sing_tid", "sing_s", "sing_y", "sing_ycont",
                     "pair_x1", "pair_x2", "pair_cid", "pair_tid", "pair_s", "pair_y", "pair_ycont", "pair_drug",
                     "pair_m", "pair_conc", "pair_dur",
                     "drug_drug", "drug_m", "drug_threshold", "labeled_pert_cellid"]
    ## check if all expected variables really are in the RData file
    assert (np.all(np.in1d(expected_keys, keys)))

    data = dict()
    for k in expected_keys:
        X = np.array(robjects.r['get'](k))
        origDim = np.array(robjects.r['dim'](robjects.r['get'](k)))
        if origDim is not None and not all(origDim == np.array(X.shape)): X = X.T
        data[k] = X
        print(k, origDim, X.dtype, X.shape)

    return data


def split_data_sd(data, selectDrug, dataMode, fold, n_folds=5, rnds=None, noPairTest=False,
                  unlab_token=-47, verbose=True):
    ''' Get data of a selected drug from the data set and split for cross-validation:
        * pair (perturbation) data:
            - split to train/valid/test by GroupKFold
            - if @noPairTest set to True, then split just to train/valid (possibly used for SSVAE, DrVAE)
        * singleton data:
            - assign cell lines that also have perturbation pair data to train/valid/test according to pair data split
            - split the rest to train/valid/test by StratifiedKFold
    '''
    #### list of drugs
    all_drugs = data['drug_drug']
    if not selectDrug in all_drugs: raise Exception("drug data not found for " + selectDrug)
    drugIndex = np.where(all_drugs == selectDrug)[0][0]
    if verbose: print(selectDrug, ": index =", drugIndex)

    #### paired perturbation data for the chosen drug
    ## "pair_x1", "pair_x2", "pair_cid", "pair_tid", "pair_s", "pair_y", "pair_ycont", "pair_drug", "pair_m", "pair_conc", "pair_dur"
    alldrugs_ppair_data = selectFromDict(data,
                                         np.asarray(list(data.keys()))[[k.startswith('pair_') for k in data.keys()]])
    ## subset to selectDrug
    all_ppairs = subsetDict(alldrugs_ppair_data, alldrugs_ppair_data['pair_drug'] == selectDrug)
    # ## select duration of ~6h
    # all_ppairs = subsetDict(all_ppairs, np.logical_and(all_ppairs['pair_dur'] > 5, all_ppairs['pair_dur'] < 7))
    ## strip 'pair_' prefix from the keys
    for k in list(all_ppairs.keys()):
        if k.startswith('pair_'): all_ppairs[k[len('pair_'):]] = all_ppairs.pop(k)
    all_ppairs['y'] = np.asarray(all_ppairs['y'], dtype=np.int32)
    # print("all_ppairs", sorted(all_ppairs.keys()))
    # for k,v in all_ppairs.items(): print(k, v.shape)

    ## denote unlabeled data as @unlab_token
    assert np.all((all_ppairs['y'] == -1) == np.isnan(all_ppairs['ycont']))
    all_ppairs['has_y'] = all_ppairs['y'] != -1
    all_ppairs['y'][np.logical_not(all_ppairs['has_y'])] = unlab_token
    all_ppairs['ycont'][np.logical_not(all_ppairs['has_y'])] = unlab_token
    if verbose:
        print("ppair labels:", Counter(all_ppairs['y']))
        print("ppair labels:", stats.describe(all_ppairs['ycont'][all_ppairs['ycont'] != unlab_token]),
              "unlabeled:", np.sum(all_ppairs['ycont'] == unlab_token))

    #### singleton sensitivity data for the chosen drug
    ## "sing_x1", "sing_cid", "sing_tid", "sing_s", "sing_y", "sing_ycont"
    all_sing = selectFromDict(data, np.asarray(list(data.keys()))[[k.startswith('sing_') for k in data.keys()]])
    ## strip 'sing_' prefix from the keys
    for k in list(all_sing.keys()):
        if k.startswith('sing_'): all_sing[k[len('sing_'):]] = all_sing.pop(k)
    ## subset to selectDrug
    # all_sing['x1'] = sklearn.preprocessing.minmax_scale(all_sing['x1']) ######################################
    # all_sing['x1'] = sklearn.preprocessing.scale(all_sing['x1']) #############################################
    # all_sing['x1'] = sklearn.preprocessing.robust_scale(all_sing['x1']) ######################################
    # scaled_tmp = sklearn.preprocessing.robust_scale(np.concatenate((all_sing['x1'], all_ppairs['x1']))) ######
    # all_sing['x1'] = scaled_tmp[:all_sing['x1'].shape[0]]
    # all_ppairs['x1'] = scaled_tmp[all_sing['x1'].shape[0]:]

    all_sing['y'] = np.asarray(all_sing['y'][:, drugIndex], dtype=np.int32)
    all_sing['ycont'] = all_sing['ycont'][:, drugIndex]
    all_sing['drug'] = np.array([data['drug_drug'][drugIndex]] * all_sing['x1'].shape[0])
    all_sing['m'] = np.array([data['drug_m'][drugIndex]] * all_sing['x1'].shape[0])
    # print("all_sing", sorted(all_sing.keys()))
    # for k,v in all_sing.items(): print(k, v.shape)

    ## denote unlabeled data as -47
    assert np.all((all_sing['y'] == -1) == np.isnan(all_sing['ycont']))
    all_sing['has_y'] = all_sing['y'] != -1
    all_sing['y'][np.logical_not(all_sing['has_y'])] = unlab_token
    all_sing['ycont'][np.logical_not(all_sing['has_y'])] = unlab_token
    if verbose:
        print("sing labels:", Counter(all_sing['y']))
        print("sing labels:", stats.describe(all_sing['ycont'][all_sing['ycont'] != unlab_token]),
              "unlabeled:", np.sum(all_sing['ycont'] == unlab_token))

    ############ Train - Valid - Test split
    ### split perturbation pairs:
    # no intersection of cell lines between the k folds
    if verbose: print("cell-line-wise Group K-Fold split on perturbation data")
    ind_tr, ind_va, ind_te = getSplitsByGroupKFold(all_ppairs['cid'], fold, n_folds, True, rnds)
    if noPairTest == False:
        ppair = subsetDict(all_ppairs, ind_tr)
        ppairv = subsetDict(all_ppairs, ind_va)
        ppairt = subsetDict(all_ppairs, ind_te)
    else:  #### no perturbation test set
        ind_tr = np.concatenate((ind_tr, ind_te))
        del ind_te
        ppair = subsetDict(all_ppairs, ind_tr)
        ppairv = subsetDict(all_ppairs, ind_va)
        ## no test set, use empty dummy with a matching dimensionality and dtype
        ppairt = dict([(k, np.empty([0] + list(v.shape[1:]), v.dtype)) for k, v in all_ppairs.items()])

    ### split singleton data:
    # dataMode = 'strictC2C' ## no patient data used; even in unlabeled data
    # dataMode = 'C2C' ## no labeled patient data used    
    # dataMode = 'C2P' ## hold out labeled patient data for testing; use unlabeled patient data in training
    # dataMode = 'CP2C' ##use patient data (both labeled and unlabeled) in training
    if dataMode in ['strictC2C']:
        ## Set aside cell lines that are also in perturbation experiments
        ## (and later add them to respective train/valid/test set)
        sing_wpert_tr = subsetDict(all_sing, np.asarray([x in ppair['cid'] for x in all_sing['cid']]))
        sing_wpert_va = subsetDict(all_sing, np.asarray([x in ppairv['cid'] for x in all_sing['cid']]))
        sing_wpert_te = subsetDict(all_sing, np.asarray([x in ppairt['cid'] for x in all_sing['cid']]))
        # only in viability screens
        unique_idx = np.logical_not(np.asarray([x in all_ppairs['cid'] for x in all_sing['cid']]))
        sing_unique = subsetDict(all_sing, unique_idx)

        ## stratified splits by response class, i.e. preserving the percentage of samples for each class in each fold
        if verbose: print("response-wise Stratified K-Fold split on singleton data")
        ind_tr, ind_va, ind_te = getSplitsByStratifiedKFold(sing_unique['y'], fold, n_folds, True, rnds)
        # apply the splits
        sing = subsetDict(sing_unique, ind_tr)
        singv = subsetDict(sing_unique, ind_va)
        singt = subsetDict(sing_unique, ind_te)

        ## merge cell lines w/perturbations to train/valid/test set
        sing = concatDictElemetwise(sing, sing_wpert_tr)
        singv = concatDictElemetwise(singv, sing_wpert_va)
        singt = concatDictElemetwise(singt, sing_wpert_te)
    else:
        raise Exception('Data mode not supported: ' + dataMode)

    return sing, singv, singt, ppair, ppairv, ppairt


def normalize01(a):
    return (a - a.min()) / (a.max() - a.min())


def makeImage(a, norm01=True, shape=None):
    a = a.reshape(-1)
    if norm01:
        a = normalize01(a)
    if shape is not None:
        img = np.zeros(shape[0] * shape[1])
        img[:len(a)] = a
        img = img.reshape([1] + list(shape))
    else:
        d = int(np.ceil(np.sqrt(len(a))))
        img = np.zeros(d ** 2)
        img[:len(a)] = a
        img = img.reshape((1, d, d))
    return np.vstack([img] * 3)
