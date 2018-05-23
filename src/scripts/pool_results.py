#!/usr/bin/env python
"""
Ladislav Rampasek (rampasek@gmail.com)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import warnings
from collections import OrderedDict
import cPickle as pickle

import numpy as np
import pandas as pd
from scipy import stats


def wilcox_test_R(x, y, alternative='less'):
    """
    Call R implementation of single-sided Wilcoxon rank sum test
    with alternative hypothesis that @x is less than @y

    NOTE: Calling R many times is slow! rather use python function if possible
    """
    if alternative not in ['two.sided', 'less', 'greater']:
        raise ValueError("Alternative hypothesis should be either 'two.sided', 'less' or 'greater'")

    import rpy2
    from rpy2.robjects.numpy2ri import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    statspackage = importr('stats', robject_translations={'format_perc': '_format_perc'})
    result = statspackage.wilcox_test(numpy2ri(x), numpy2ri(y), alternative=alternative,
                                      paired=True, exact=False, correct=False)

    pyresultdict = pandas2ri.ri2py(result)
    for k, v in pyresultdict.items():
        # print(k, v)
        if k == 'p.value':
            pval = v[0]
    return pval


def my_wilcoxon_test(x, y=None, alternative='less', correction=False):
    """
    Calculate the paired Wilcoxon signed-rank test.

    ** Modified scipy implementation to mimic R implementation with support for one-sided tests **

    https://github.com/scipy/scipy/blob/v1.0.0/scipy/stats/morestats.py#L2316-L2413
    https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/wilcox.test.R
    
    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    alternative : string, {"two.sided", "less", "greater"}, optional
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.
    Returns
    -------
    float: The single-sided p-value for the test.
    
    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    """

    if alternative not in ['two.sided', 'less', 'greater']:
        raise ValueError("Alternative hypothesis should be either 'two.sided', 'less' or 'greater'")

    if y is None:
        d = asarray(x)
    else:
        x, y = map(np.asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x - y

    # Keep all non-zero differences (zero_method == "wilcox")
    d = np.compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 20:
        warnings.warn("Warning: sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    T = np.sum((d > 0) * r, axis=0)

    mn = count * (count + 1.) / 4.
    se = count * (count + 1.) * (2. * count + 1.) / 24.

    replist, repnum = stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= (repnum ** 3 - repnum).sum() / 48.

    se = np.sqrt(se)

    correct = 0.
    if correction:
        if alternative == "two.sided":
            correct = 0.5 * np.sign(T - mn)
        elif alternative == "greater":
            correct = 0.5
        elif alternative == "less":
            correct = -0.5

    z = (T - mn - correct) / se

    prob = None
    if alternative == "two.sided":
        prob = 2. * min(stats.distributions.norm.cdf(z), stats.distributions.norm.sf(z))
    elif alternative == "greater":
        prob = stats.distributions.norm.sf(z)
    elif alternative == "less":
        prob = stats.distributions.norm.cdf(z)

    return prob


def get_select_map(stat_names, tasks=None, models=None, measure=None):
    choice = list()
    for stat in stat_names:
        if len(stat.split('|')) >= 3:
            selected = True
            for st in stat.split(';'):
                parts = st.split('|')
                if tasks is not None:
                    if parts[0] not in tasks:
                        selected = False
                if measure is not None:
                    if parts[1] not in measure:
                        selected = False
                if models is not None:
                    if parts[2] not in models:
                        selected = False
            if selected:
                choice.append(stat)
    choice = sorted(choice)
    return choice


def to_table(s, measure="AUROC", tasks=None, models=None, drugs=None):
    drug_names = list(s.keys())
    stat_names = list(s[s.keys()[0]].keys())

    assert drugs is not None
    # drug_list = [dn for dn in drug_names if dn in drugs]
    drug_list = drugs

    choice = get_select_map(stat_names, tasks, models, measure)

    choice_names = list()
    for stat in choice:
        stat_split = stat.split('|')
        new_name = ''
        if len(stat_split) == 3:
            new_name = stat_split[0] + '|' + stat_split[2]
        else:
            new_name = stat_split[0] + '|' + stat_split[2] + '|' + stat_split[4]
        choice_names.append(new_name)
    # choice_names = [stat.split('|')[0] + '|' + stat.split('|')[2] for stat in choice]

    compare_stats = []
    compare_stats_names = [choice[cs[0]] + ' < ' + choice[cs[1]] for cs in compare_stats]

    first_line = ['drug'] + choice_names + compare_stats_names
    table = [first_line]
    for drug in drug_list:
        if not drug in s:
            warnings.warn('Warning! ignoring ' + drug + ' (no stats)')
            continue
        line = [drug]
        for sn in choice:
            line.append(s[drug][sn])
        for cs in compare_stats:  # compare to the baseline
            line.append(int(s[drug][choice[cs[0]]] < s[drug][choice[cs[1]]]))
        table.append(line)
    line = ['MEAN']
    line.extend(np.asarray([l[1:] for l in table[1:]], dtype=float).mean(0).tolist())
    table.append(line)

    pdtable = pd.DataFrame(table[1:])
    pdtable.columns = first_line
    pdtable = pdtable.rename(dict([(i, n) for i, n in enumerate(drug_list + ['MEAN'])]))

    # print(dict([(i,n) for i,n in enumerate(drug_list)]))
    # pd.options.display.float_format = '{:.3f}'.format
    # print(pdtable)

    return table, pdtable


def compute_pairwise_pvals(rlist, tasks, models):
    res = dict()
    num = len(rlist)
    for drug in rlist[0].keys():
        if not np.all(np.array([drug in rl.keys() for rl in rlist])):
            warnings.warn('Warning! ignoring ' + drug + ' as it is not in all input files')
            continue

        dv = dict()
        for ref_task in tasks:
            for ref_model in models:
                for key in rlist[0][drug].keys():
                    # ignore entries that are not performance stats
                    if '|' not in key: continue
                    if '->Y' not in key: continue

                    # compute statistical test between this classification result and our DGM
                    measure = key.split('|')[1]
                    ref_key = ref_task + '|' + measure + '|' + ref_model
                    if ref_key not in rlist[0][drug].keys():
                        # print("skipping non existent reference " + ref_key)
                        continue
                    if ref_key == key:  # don't compare to self
                        continue

                    vals = list()
                    ref_vals = list()
                    # get list of results
                    for rl in rlist:
                        try:
                            vals.append(rl[drug][key][0] if isinstance(rl[drug][key], list) else rl[drug][key])
                            ref_vals.append(rl[drug][ref_key][0] if isinstance(rl[drug][ref_key], list) else rl[drug][ref_key])
                        except:
                            warnings.warn('Warning! missing entry for ' + drug + ' ' + key)
                            pass

                    vals = np.asarray(vals)

                    ref_vals = np.asarray(ref_vals)
                    ## scipy implementation of *two-sided* Wilcoxon test
                    # dv[key] = stats.wilcoxon(ref_vals, vals)[1]
                    ## call R implementation of *one-sided* Wilcoxon test
                    # dv[key] = wilcox_test_R(vals, ref_vals, alternative="less")
                    ## calling R is too slow, run my reimplemented Wilcoxon test
                    # print(ref_key+';'+key, len(vals), len(ref_vals))
                    dv[ref_key+';'+key] = my_wilcoxon_test(vals, ref_vals, alternative="less")

        res[drug] = dv
    return res


def compute_result_stat(rlist, report_stat='mean'):
    res = dict()
    num = len(rlist)
    for drug in rlist[0].keys():
        if not np.all(np.array([drug in rl.keys() for rl in rlist])):
            warnings.warn('Warning! ignoring ' + drug + 'as it is not in all input files')
            continue
        dv = dict()
        for key in rlist[0][drug].keys():
            # ignore entries that are not performance stats
            if '|' not in key: continue

            # compute statistical test between this classification result and our DGM
            compare_to_ref = '->Y' in key
            if compare_to_ref:
                measure = key.split('|')[1]
                ref_key = 'X1->Y|' + measure + '|DGM'
                ref_vals = list()
                if ref_key == key:  # don't compare to self
                    compare_to_ref = False

            vals = list()
            # get list of results
            for rl in rlist:
                try:
                    vals.append(rl[drug][key][0] if isinstance(rl[drug][key], list) else rl[drug][key])
                    if compare_to_ref:
                        ref_vals.append(rl[drug][ref_key][0] if isinstance(rl[drug][ref_key], list) else rl[drug][ref_key])
                except:
                    warnings.warn('Warning! missing entry for ' + drug + ' ' + key)
                    pass

            vals = np.asarray(vals)

            if report_stat == 'cfint':
                cf95plusminus = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
                # cf95plusminus = 1.96 * vals[np.isnan(vals) == False].std(ddof=1) / np.sqrt((np.isnan(vals) == False).sum())
                dv[key] = cf95plusminus

            elif report_stat == 'wilcoxon':
                if compare_to_ref:
                    ref_vals = np.asarray(ref_vals)
                    ## scipy implementation of *two-sided* Wilcoxon test
                    # dv[key] = stats.wilcoxon(ref_vals, vals)[1]
                    ## call R implementation of *one-sided* Wilcoxon test
                    # dv[key] = wilcox_test_R(vals, ref_vals, alternative="less")
                    ## calling R is too slow, run my reimplemented Wilcoxon test
                    dv[key] = my_wilcoxon_test(vals, ref_vals, alternative="less")

                    # print(ref_vals.mean(), vals.mean())
                    # print("R", wilcox_test_R(vals, ref_vals, alternative="less"))
                    # print("my", my_wilcoxon_test(vals, ref_vals, alternative="less"))
                else:
                    dv[key] = np.mean(vals)

            elif report_stat == 'mean':
                dv[key] = np.mean(vals)
                if np.isnan(dv[key]):
                    warnings.warn('Warning! NaN in result values')
                    # vals[np.isnan(vals)] = 1.
                    # dv[key] = np.mean(vals)
                    dv[key] = np.nanmean(vals)

            else:
                raise ValueError('Unknown statistic: ' + report_stat)

            # conf_interval95 = stats.norm.interval(0.95, loc=vals.mean(), scale=vals.std(ddof=1)/np.sqrt(len(vals)))
            # print((conf_interval95[1] - conf_interval95[0])/2, 1.96*vals.std(ddof=1)/np.sqrt(len(vals)), len(vals))

            ## confidence interval with t-distrib # with 50 samples it is very similar to that of normal distrib.
            # # Get the endpoints of the range that contains 95% of the distribution
            # t_bounds = stats.t.interval(0.95, len(vals) - 1)
            # # sum mean to the confidence interval
            # ci = [vals.mean() + critval * vals.std(ddof=1) / np.sqrt(len(vals)) for critval in t_bounds]
            # print ("Mean: {}".format(vals.mean()))
            # print ("Confidence Interval 95%: {}, {}".format(ci[0], ci[1]))
        res[drug] = dv
    return res


def load_from_jason(fname):
    """Load data from a JASON file to a dictionary."""
    with open(fname, 'rb') as f:
        res = OrderedDict(json.load(f))
    return res


def main(files, measures, tasks=None, models=None, drugs=None, sep='tab', report_stat='mean',
         report_valid=False, dump_rlist=False):
    print('Number of files on input: ', len(files))
    ## load JSON files
    rlist = []
    for fn in files:
        r = load_from_jason(fn)
        rlist.append(r)

    ## prepare experiment ID such that we average over some parameters and make validation-set selection over other
    ## e.g. average over random seeds and CV folds and choose the best Y-loss rate on validation set
    eids = []
    for fn in files:
        hparam_fields = os.path.splitext(os.path.basename(fn))[0].split('_')
        selection_hparams = list()
        for hp in hparam_fields:
            ## pool based on these fields: RS - random seed; FOLD - CV fold
            if hp.startswith('RS') or hp.startswith('FOLD'):
                continue
            ## make an ID from hyperparameters to do selection over
            selection_hparams.append(hp)
        idstr = '_'.join(selection_hparams)
        eids.append(idstr)
    eids = np.asarray(eids)
    masterids = np.unique(eids)
    print("Unique IDs:", masterids)

    test_mids = sorted([mid for mid in masterids if 'test' in mid])
    valid_mids = sorted([mid for mid in masterids if 'valid' in mid])
    assert len(test_mids) == 1 or len(valid_mids) == 1 or (len(test_mids) == len(valid_mids))

    pooled_results = dict()
    for mid in masterids:
        tmp_rlist = list()
        for i, eid in enumerate(eids):
            if eid == mid: tmp_rlist.append(rlist[i])
        if mid in test_mids:
            if report_stat == 'wilcoxon-allvsall':
                pooled_results[mid] = compute_pairwise_pvals(tmp_rlist, tasks, models)
            else:   
                pooled_results[mid] = compute_result_stat(tmp_rlist, report_stat)
        else:  # on validation results compute mean performance for model selection
            pooled_results[mid] = compute_result_stat(tmp_rlist, 'mean')

    ## select drugs
    drug_list = sorted(pooled_results[masterids[0]].keys())
    if drugs is not None and drugs[0] not in ['all', '26']:
        drug_list = [dn for dn in drugs if dn in drug_list]
    else:
        drug_list = ['bortezomib', 'clofarabine', 'dasatinib', 'decitabine', 'docetaxel', 'etoposide', 'gemcitabine', 'mitomycin', 'paclitaxel', 'PLX-4032', 'topotecan', 'vincristine', 'vorinostat']
        if drugs is not None and drugs[0] == '26':
            drug_list = ['omacetaxine mepesuccinate', 'bortezomib', 'vorinostat', 'paclitaxel', 'docetaxel', 'topotecan', 'niclosamide', 'valdecoxib','teniposide', 'vincristine', 'prochlorperazine', 'mitomycin', 'lovastatin', 'gemcitabine', 'dasatinib', 'fluvastatin', 'clofarabine', 'sirolimus', 'etoposide', 'sitagliptin', 'decitabine', 'PLX-4032', 'fulvestrant', 'bosutinib', 'trifluoperazine', 'ciclosporin']
        # drug_list = sorted(drug_list)
        if drugs is not None and drugs[0] == 'all':
            for dd in sorted(pooled_results[masterids[0]].keys()):
                if dd not in drug_list:
                    drug_list += [dd]

    if len(test_mids) + len(valid_mids) == 1:
        ## if only one validation or test hyper parameter setting
        if len(test_mids) == 1:
            chosen_result = pooled_results[test_mids[0]]
        elif len(valid_mids) == 1:
            chosen_result = pooled_results[valid_mids[0]]
        for ms in measures:
            print(ms)
            _, pdtable = to_table(chosen_result, ms, tasks, models, drug_list)
            if sep == 'tex':
                print(pdtable.to_csv(sep='&', index=False, float_format='%.3f'))
            elif sep == 'csv':
                print(pdtable.to_csv(sep=',', index=False, float_format='%.3E'))
            elif sep == 'tab':
                print('\t' + pdtable.to_string(index=False, float_format='%.3f'))
            else:
                raise ValueError('Unknown separator type ', sep)
        # dump the rlists:
        if dump_rlist is not None:
            stat_names = list(rlist[0].values()[0].keys())
            choice = get_select_map(stat_names, tasks, models, measures)
            selected_rlists = dict()
            for d in drug_list:
                selected_rlists[d] = list()
                for j in range(len(rlist)):
                    selected_rlists[d].append({d: dict()})
                    for stat in choice:
                        stat_split = stat.split('|')
                        if len(dump_rlist) > 0: stat_split[-1] = dump_rlist + '-' + stat_split[-1]
                        new_name = '|'.join(stat_split)
                        selected_rlists[d][-1][d][new_name] = rlist[j][d][stat]

            with open(files[0].split('/')[0]+'-rlist.pkl', 'wb') as f:
                pickle.dump(selected_rlists, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("SAVED raw test result dictionaries obtained with the selected hyperparams")
    else:
        ## select best hyper parameter settings for each drug on validation and report test performance
        if 'AUROC' in measures:
            selection_ms = 'AUROC'
            selection_task = 'X1->Y'
        else:
            selection_ms = 'PearR'
            selection_task = 'PERT_X2'

        # compute validation performance of our DGM on the prediction task for all settings
        valid_tables = dict()
        for mid in valid_mids:
            _, pdt = to_table(pooled_results[mid], selection_ms, [selection_task], ['DGM'], drug_list)
            valid_tables[mid] = pdt

        ## select IDs of the best setting for each drug based on validation performance
        selected_rlists = dict()
        selected_mids = dict()
        for d in drug_list:
            best_valid_val = -999.
            best_valid_mid = None
            for mid, table in valid_tables.items():
                x = float(table.loc[[d]][selection_task + '|DGM'])
                if x > best_valid_val:
                    best_valid_val = x
                    best_valid_mid = mid
            if not report_valid:
                ## report performance on test set
                chosen_test_mid = best_valid_mid.replace('valid', 'test')
                assert chosen_test_mid in test_mids
            else:
                ## report validation performance
                print("!!! RETURNING VALIDATION PERFORMANCE !!!")
                chosen_test_mid = best_valid_mid
                test_mids = valid_mids
            selected_mids[d] = chosen_test_mid
            print(d, best_valid_val, best_valid_mid, chosen_test_mid)
            # prepare for saving raw test result dictionaries obtained with the selected hyperparams
            if dump_rlist is not None:
                tmp_rlist = list()
                for i, eid in enumerate(eids):
                    if eid == chosen_test_mid: tmp_rlist.append(rlist[i])
                assert chosen_test_mid in test_mids

                stat_names = list(rlist[0].values()[0].keys())

                choice = get_select_map(stat_names, tasks, models, measures)
                selected_rlists[d] = list()
                for j in range(len(tmp_rlist)):
                    selected_rlists[d].append({d: dict()})
                    for stat in choice:
                        stat_split = stat.split('|')
                        if len(dump_rlist) > 0: stat_split[-1] = dump_rlist + '-' + stat_split[-1]
                        new_name = '|'.join(stat_split)
                        selected_rlists[d][-1][d][new_name] = tmp_rlist[j][d][stat]
        # dump the selected_rlists:
        if dump_rlist is not None:
            with open(files[0].split('/')[0]+'-rlist.pkl', 'wb') as f:
                pickle.dump(selected_rlists, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("SAVED raw test result dictionaries obtained with the selected hyperparams")

        ## report test performance of selected hyperparameter settings
        for ms in measures:
            print(ms)
            test_tables = dict()
            for mid in test_mids:
                _, pdt = to_table(pooled_results[mid], ms, tasks, models, drug_list)
                test_tables[mid] = pdt
            pdtable = test_tables[test_mids[0]].copy()
            for d in drug_list:
                # take results for drug d computed using the selected hyperparameter settings and
                # place them to the final summary table
                pdtable.loc[[d]] = test_tables[selected_mids[d]].loc[[d]].copy()

            # recompute means of the columns
            new_means = pdtable.drop('drug', axis=1).apply(lambda x: x.mean())
            new_means.loc['drug'] = 'MEAN'
            pdtable.loc['MEAN'] = new_means

            # print(pdtable)
            if sep == 'tex':
                print(pdtable.to_csv(sep='&', index=False, float_format='%.3f'))
            elif sep == 'csv':
                print(pdtable.to_csv(sep=',', index=False, float_format='%.3E'))
            elif sep == 'tab':
                print('\t' + pdtable.to_string(index=False, float_format='%.3f'))
            else:
                raise ValueError('Unknown separator type ', sep)


if __name__ == '__main__':
    ### Parse command line arguments
    parser = argparse.ArgumentParser(description='Results averaging: Drug response prediction VAE model.')
    parser.add_argument('--files', nargs='+', type=str, default=None)
    parser.add_argument('--measure', nargs='+', type=str, default=['AUPR', 'AUROC', 'Acc'], required=False,
                        help='Specify which measures to report')
    parser.add_argument('--task', nargs='+', type=str, default=None,  # ['X1->Y','Z1->Y'],
                        required=False, help='Filter which tasks to report e.g. "X1->Y" (default is all)')
    parser.add_argument('--model', nargs='+', type=str, default=None,  # ['DGM','Ridge'],
                        required=False, help='Filter which models to report e.g. "DGM" (default is all)')
    parser.add_argument('--drug', nargs='+', type=str, default=None, required=False,
                        help='Filter which drugs to report e.g. "PLX-4032" (default is all)')
    parser.add_argument('--sep', type=str, default='tab', required=False,
                        help='Output format (tex, csv, tab) default: tab')
    parser.add_argument('--cf-interval', action='store_true', default=False,
                        help='Report 95%% confidence interval +/- from the mean')
    parser.add_argument('--wilcoxon', action='store_true', default=False,
                        help='Report Wilcoxon signed-rank test p-value of X1->Y|*|DGM > baseline')
    parser.add_argument('--wilcoxon-allvsall', action='store_true', default=False,
                        help='Report Wilcoxon signed-rank test p-value of all vs all')
    parser.add_argument('--valid', action='store_true', default=False,
                        help='Report validation set performance instead of test')
    parser.add_argument('--dump-rlist', type=str, default=None,
                        help='Store result lists to a pickle file. These are used for comaprison between DrVAE/SSVAE \
                        models by compare_pooled_results.py script.')

    args = parser.parse_args()
    # print(args)

    assert (args.sep in ('tex', 'csv', 'tab')), 'Unknown format type'
    assert not (args.wilcoxon and args.cf_interval and args.wilcoxon_allvsall), 'given report stats are mutually exclusive'
    report_stat = 'mean'
    if args.cf_interval:
        report_stat = 'cfint'
    elif args.wilcoxon:
        report_stat = 'wilcoxon'
    elif args.wilcoxon_allvsall:
        report_stat = 'wilcoxon-allvsall'

    main(files=args.files, measures=args.measure, tasks=args.task, models=args.model, drugs=args.drug,
         sep=args.sep, report_stat=report_stat, report_valid=args.valid, dump_rlist=args.dump_rlist)
