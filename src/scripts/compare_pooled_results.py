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
import copy
import warnings
from collections import OrderedDict
import cPickle as pickle
from pprint import pprint

import numpy as np
import pandas as pd
from scipy import stats


def wilcox_test_R(x, y, alternative='less', paired=True):
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
                                      paired=paired, exact=False, correct=False)

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

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            z = (T - mn - correct) / se
        except Warning as e:
            z = 0.
            # print(T, mn, correct, se)
            # print(e)

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
            # for st in stat.split(';'):
            st = stat.split(';')[0]
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

    first_line = ['drug'] + choice_names
    table = [first_line]
    for drug in drug_list:
        if not drug in s:
            print('Warning! ignoring', drug, '(no stats)')
            continue
        line = [drug]
        for sn in choice:
            line.append(s[drug][sn])
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


def compute_pairwise_pvals(rlist, measures, tasks, models):
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
                    if measure not in measures: continue
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


def compute_result_stat(rlist, measures, report_stat='mean', paired=None, cmpto=None):
    res = dict()
    num = len(rlist)
    for drug in rlist[0].keys():
        if not np.all(np.array([drug in rl.keys() for rl in rlist])):
            print('Warning! ignoring', drug, 'as it is not in all input files')
            continue
        dv = dict()
        for key in rlist[0][drug].keys():
            # ignore entries that are not performance stats
            if '|' not in key: continue

            # compute statistical test between this classification result and the "DGM2"
            measure = key.split('|')[1]
            if measure not in measures: continue
            compare_to_ref = '->Y' in key and cmpto is not None
            if compare_to_ref:
                cmp_data = cmpto.split('|')[0] 
                cmp_model = cmpto.split('|')[1]
                ref_key = cmp_data + '|' + measure + '|' + cmp_model
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
                    print('Warning! missing entry for ', drug, key, ref_key)
                    pass

            vals = np.asarray(vals)

            if report_stat == 'cfint':
                cf95plusminus = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
                dv[key] = cf95plusminus

            elif report_stat == 'wilcoxon':
                if compare_to_ref:
                    ref_vals = np.asarray(ref_vals)
                    ## scipy implementation of *two-sided* Wilcoxon test
                    # dv[key] = stats.wilcoxon(ref_vals, vals)[1]
                    if not paired:
                        ## call R implementation of *one-sided* Wilcoxon test
                        dv[key] = wilcox_test_R(ref_vals, vals, alternative="less", paired=paired)
                        # print(drug, key, dv[key])
                    else:
                        ## calling R is too slow, run my reimplementation of paired Wilcoxon test
                        ## equivalent to calling wilcox_test_R(ref_vals, vals, alternative="less", paired=True)
                        dv[key] = my_wilcoxon_test(ref_vals, vals, alternative="less")

                    # print(ref_vals.mean(), vals.mean())
                    # print("R", wilcox_test_R(vals, ref_vals, alternative="less"))
                    # print("my", my_wilcoxon_test(vals, ref_vals, alternative="less"))
                else:
                    dv[key] = np.mean(vals)

            elif report_stat == 'mean':
                dv[key] = np.mean(vals)

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


def main(files, measures, tasks=None, models=None, drugs=None, sep='tab', report_stat='mean', paired_test=None,
         report_valid=False, cmpto=None):
    print('Number of files on input: ', len(files))
    twofile_comparison = bool(report_stat == 'wilcoxon')
    if twofile_comparison:
        assert len(files) == 2

    ## load result lists from pickle
    rlist = []
    for fn in files:
        with open(fn, 'rb') as f:
            r = pickle.load(f)
        rlist.append(r)

    # pprint(rlist[0], depth=1)
    # pprint(rlist[0]['bortezomib'][0].keys())

    firstrl = rlist[0]
    joint_rlist = copy.deepcopy(firstrl.values()[0])

    for d in firstrl.keys():
        for i in range(len(firstrl[d])):            
            # pprint(firstrl[d][i][d])
            joint_rlist[i][d] = firstrl[d][i][d]
            
            # if specified model to compare to, select only that one from the other file(s)
            if cmpto is not None:
                cmp_data = cmpto.split('|')[0] 
                cmp_model = cmpto.split('|')[1]
            else:
                cmp_data = cmp_model = ''

            ## create joint results list by appending results from the other result lists to the first list
            for nextrl in rlist[1:]:
                assert len(firstrl[d]) == len(nextrl[d])
                for experiment in nextrl[d][i][d].keys():
                    if cmp_data in experiment and cmp_model in experiment:
                        # experiment_mod = experiment.replace(cmp_model, 'list2-'+cmp_model)
                        joint_rlist[i][d][experiment] = nextrl[d][i][d][experiment]
    
    # if no models were specified, compute stats for all
    if models is None:
        models = [key.split('|')[2] for key in joint_rlist[0].values()[0].keys()]

    if report_stat == 'wilcoxon-allvsall':
        results = compute_pairwise_pvals(joint_rlist, measures, tasks, models)
    else:   
        results = compute_result_stat(joint_rlist, measures, report_stat, paired_test, cmpto)

    ## select drugs
    drug_list = sorted(joint_rlist[0].keys())
    if drugs is not None and drugs[0] not in ['all', '26']:
        drug_list = [dn for dn in drugs if dn in drug_list]
    else:
        drug_list = ['bortezomib', 'clofarabine', 'dasatinib', 'decitabine', 'docetaxel', 'etoposide', 'gemcitabine', 'mitomycin', 'paclitaxel', 'PLX-4032', 'topotecan', 'vincristine', 'vorinostat']
        if drugs is not None and drugs[0] == '26':
            drug_list = ['omacetaxine mepesuccinate', 'bortezomib', 'vorinostat', 'paclitaxel', 'docetaxel', 'topotecan', 'niclosamide', 'valdecoxib','teniposide', 'vincristine', 'prochlorperazine', 'mitomycin', 'lovastatin', 'gemcitabine', 'dasatinib', 'fluvastatin', 'clofarabine', 'sirolimus', 'etoposide', 'sitagliptin', 'decitabine', 'PLX-4032', 'fulvestrant', 'bosutinib', 'trifluoperazine', 'ciclosporin']
        # drug_list = sorted(drug_list)
        if drugs is not None and drugs[0] == 'all':
            for dd in sorted(joint_rlist[0].keys()):
                if dd not in drug_list:
                    drug_list += [dd]

    if cmpto is not None:
        models = models + [cmp_model]

    for ms in measures:
        print(ms)
        _, pdtable = to_table(results, ms, tasks, models, drug_list)
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
    parser.add_argument('--wilcoxon-paired', action='store_true', default=False,
                        help='Report paired Wilcoxon signed-rank test p-value of X1->Y|*|DrVAE-DGM > X1->Y|*|SSVAE-DGM')
    parser.add_argument('--wilcoxon-2sample', action='store_true', default=False,
                        help='Report unpaired 2-sample Wilcoxon signed-rank test p-value of X1->Y|*|DrVAE-DGM > X1->Y|*|SSVAE-DGM')
    parser.add_argument('--wilcoxon-allvsall', action='store_true', default=False,
                        help='Report Wilcoxon signed-rank test p-value of all vs all, concatenating all result list files on input')
    parser.add_argument('--valid', action='store_true', default=False,
                        help='Report validation set performance instead of test')
    parser.add_argument('--cmpto', type=str, default=None,
                        help='what model from the second file to select and compare to all results in first file, e.g. X1->Y|SSVAE-DGM')

    args = parser.parse_args()
    # print(args)

    assert (args.sep in ('tex', 'csv', 'tab')), 'Unknown format type'
    assert not (args.wilcoxon_paired and args.wilcoxon_2sample and args.cf_interval and args.wilcoxon_allvsall), \
        '--cf-interval, --wilcoxon-paired, --wilcoxon-2sample and --wilcoxon-allvsall are mutually exclusive'
    
    report_stat = 'mean'
    paired_test = None
    if args.cf_interval:
        report_stat = 'cfint'
    elif args.wilcoxon_paired:
        report_stat = 'wilcoxon'
        paired_test = True
    elif args.wilcoxon_2sample:
        report_stat = 'wilcoxon'
        paired_test = False
    elif args.wilcoxon_allvsall:
        report_stat = 'wilcoxon-allvsall'
        paired_test = True

    if report_stat == 'wilcoxon':
        assert args.cmpto is not None

    main(files=args.files, measures=args.measure, tasks=args.task, models=args.model, drugs=args.drug,
         sep=args.sep, report_stat=report_stat, paired_test=paired_test, report_valid=args.valid, cmpto=args.cmpto)
