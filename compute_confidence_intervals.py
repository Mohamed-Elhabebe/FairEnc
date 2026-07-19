import os
import numpy as np
import argparse
from sklearn.utils import resample

import sys
sys.path.append('.')

from src.modules import *

parser = argparse.ArgumentParser(description='FairEnc Compute Confidence Intervals')

parser.add_argument('--exp_type', default='zero_shot', type=str)
parser.add_argument('--base_folder', default='', type=str)
parser.add_argument('--exp_index', default=0, type=int)
parser.add_argument('--n_bootstrap', default=1000, type=int)
parser.add_argument('--seed', default=0, type=int)

def analyze_npz_results_files(files, n_bootstrap = 1000):
    overall_aucs = []
    overall_es_aucs = []
    overall_aucs_by_attrs = []
    overall_dpds = []
    overall_eods = []
    
    bootstrap_aucs = []
    bootstrap_es_aucs = []
    bootstrap_aucs_by_attrs = []
    bootstrap_dpds = []
    bootstrap_eods = []

    for file in files:
        data = np.load(file)
        preds = data['val_pred']
        gts = data['val_gt']
        attrs = data['val_attr']

        overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(preds, gts, attrs.T)
        overall_aucs.append(overall_auc)
        overall_es_aucs.append(eval_es_auc)
        overall_aucs_by_attrs.append(eval_aucs_by_attrs)
        overall_dpds.append(eval_dpds)
        overall_eods.append(eval_eods)

        strata = np.array([f"{y}_{r}_{g}_{e}_{l}" for y, r, g, e, l in zip(gts, attrs[:, 0], attrs[:, 1], attrs[:, 2], attrs[:, 3])])
        
        for _ in range(n_bootstrap):
            boot_gts, boot_preds, boot_attrs = resample(
                gts, preds, attrs,
                replace=True,
                n_samples=len(gts),
                stratify=strata
            )
            bootstrap_acc, bootstrap_eval_es_acc, bootstrap_auc, bootstrap_eval_es_auc, bootstrap_eval_aucs_by_attrs, bootstrap_eval_dpds, bootstrap_eval_eods, bootstrap_between_group_disparity = evalute_comprehensive_perf(boot_preds, boot_gts, boot_attrs.T)
            bootstrap_aucs.append(bootstrap_auc)
            bootstrap_es_aucs.append(bootstrap_eval_es_auc)
            bootstrap_aucs_by_attrs.append(bootstrap_eval_aucs_by_attrs)
            bootstrap_dpds.append(bootstrap_eval_dpds)
            bootstrap_eods.append(bootstrap_eval_eods)

    overall_aucs = np.array(overall_aucs)
    overall_es_aucs = np.array(overall_es_aucs)
    overall_dpds = np.array(overall_dpds)
    overall_eods = np.array(overall_eods)

    overall_attrs_aucs = [[], [], [], []]
    for i in range(4):
        for attr_aucs in overall_aucs_by_attrs:
            overall_attrs_aucs[i].append(attr_aucs[i])
        overall_attrs_aucs[i] = np.array(overall_attrs_aucs[i])
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    bootstrap_es_aucs = np.array(bootstrap_es_aucs)
    bootstrap_dpds = np.array(bootstrap_dpds)
    bootstrap_eods = np.array(bootstrap_eods)

    bootstrap_attrs_aucs = [[], [], [], []]
    for i in range(4):
        for attr_aucs in bootstrap_aucs_by_attrs:
            bootstrap_attrs_aucs[i].append(attr_aucs[i])
        bootstrap_attrs_aucs[i] = np.array(bootstrap_attrs_aucs[i])
    
    if np.mean(overall_attrs_aucs[0][:, 0]) < np.mean(overall_attrs_aucs[0][:, 1]) and np.mean(overall_attrs_aucs[0][:, 0]) < np.mean(overall_attrs_aucs[0][:, 2]):
        worst_mean = np.mean(overall_attrs_aucs[0][:, 0])
        worst_std = np.std(overall_attrs_aucs[0][:, 0])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[0][:, 0]
    elif np.mean(overall_attrs_aucs[0][:, 1]) < np.mean(overall_attrs_aucs[0][:, 0]) and np.mean(overall_attrs_aucs[0][:, 1]) < np.mean(overall_attrs_aucs[0][:, 2]):
        worst_mean = np.mean(overall_attrs_aucs[0][:, 1])
        worst_std = np.std(overall_attrs_aucs[0][:, 1])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[0][:, 1]
    else:
        worst_mean = np.mean(overall_attrs_aucs[0][:, 2])
        worst_std = np.std(overall_attrs_aucs[0][:, 2])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[0][:, 2]
    print('Race: ', round(np.mean(overall_dpds[:, 0]) * 100, 2), ' ± ', round(np.std(overall_dpds[:, 0]) * 100, 2), ' & ', round(np.mean(overall_eods[:, 0]) * 100, 2), ' ± ', round(np.std(overall_eods[:, 0]) * 100, 2), ' & ',
                    round(np.mean(overall_aucs) * 100, 2), ' ± ', round(np.std(overall_aucs) * 100, 2), ' & ', round(np.mean(overall_es_aucs[:, 0]) * 100, 2), ' ± ', round(np.std(overall_es_aucs[:, 0]) * 100, 2), ' & ',
                    round(np.mean(overall_attrs_aucs[0][:, 0]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[0][:, 0]) * 100, 2), ' & ', round(np.mean(overall_attrs_aucs[0][:, 1]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[0][:, 1]) * 100, 2), ' & ',
                    round(np.mean(overall_attrs_aucs[0][:, 2]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[0][:, 2]) * 100, 2), ' & ', round(worst_mean * 100, 2), ' ± ', round(worst_std * 100, 2))

    print('Race: ', '(', round(np.percentile(bootstrap_dpds[:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_dpds[:, 0], 97.5) * 100, 2), ')', ' & ', 
                    '(', round(np.percentile(bootstrap_eods[:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_eods[:, 0], 97.5) * 100, 2), ')', ' & ',
                    '(', round(np.percentile(bootstrap_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_aucs, 97.5) * 100, 2), ')', ' & ', 
                    '(', round(np.percentile(bootstrap_es_aucs[:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_es_aucs[:, 0], 97.5) * 100, 2), ')', ' & ',
                    '(', round(np.percentile(bootstrap_attrs_aucs[0][:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[0][:, 0], 97.5) * 100, 2), ')', ' & ', 
                    '(', round(np.percentile(bootstrap_attrs_aucs[0][:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[0][:, 1], 97.5) * 100, 2), ')', ' & ',
                    '(', round(np.percentile(bootstrap_attrs_aucs[0][:, 2], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[0][:, 2], 97.5) * 100, 2), ')', ' & ', 
                    '(', round(np.percentile(bootstrap_worst_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_worst_aucs, 97.5) * 100, 2), ')')

    if np.mean(overall_attrs_aucs[1][:, 0]) < np.mean(overall_attrs_aucs[1][:, 1]):
        worst_mean = np.mean(overall_attrs_aucs[1][:, 0])
        worst_std = np.std(overall_attrs_aucs[1][:, 0])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[1][:, 0]
    else:
        worst_mean = np.mean(overall_attrs_aucs[1][:, 1])
        worst_std = np.std(overall_attrs_aucs[1][:, 1])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[1][:, 1]
    print('Gender: ', round(np.mean(overall_dpds[:, 1]) * 100, 2), ' ± ', round(np.std(overall_dpds[:, 1]) * 100, 2), ' & ', round(np.mean(overall_eods[:, 1]) * 100, 2), ' ± ', round(np.std(overall_eods[:, 1]) * 100, 2), ' & ',
                      round(np.mean(overall_aucs) * 100, 2), ' ± ', round(np.std(overall_aucs) * 100, 2), ' & ', round(np.mean(overall_es_aucs[:, 1]) * 100, 2), ' ± ', round(np.std(overall_es_aucs[:, 1]) * 100, 2), ' & ',
                      round(np.mean(overall_attrs_aucs[1][:, 0]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[1][:, 0]) * 100, 2), ' & ', round(np.mean(overall_attrs_aucs[1][:, 1]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[1][:, 1]) * 100, 2), ' & ',
                      round(worst_mean * 100, 2), ' ± ', round(worst_std * 100, 2))

    print('Gender: ', '(', round(np.percentile(bootstrap_dpds[:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_dpds[:, 1], 97.5) * 100, 2), ')', ' & ', 
                      '(', round(np.percentile(bootstrap_eods[:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_eods[:, 1], 97.5) * 100, 2), ')', ' & ',
                      '(', round(np.percentile(bootstrap_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_aucs, 97.5) * 100, 2), ')', ' & ', 
                      '(', round(np.percentile(bootstrap_es_aucs[:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_es_aucs[:, 1], 97.5) * 100, 2), ')', ' & ',
                      '(', round(np.percentile(bootstrap_attrs_aucs[1][:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[1][:, 0], 97.5) * 100, 2), ')', ' & ', 
                      '(', round(np.percentile(bootstrap_attrs_aucs[1][:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[1][:, 1], 97.5) * 100, 2), ')', ' & ',
                      '(', round(np.percentile(bootstrap_worst_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_worst_aucs, 97.5) * 100, 2), ')')

    if np.mean(overall_attrs_aucs[2][:, 0]) < np.mean(overall_attrs_aucs[2][:, 1]):
        worst_mean = np.mean(overall_attrs_aucs[2][:, 0])
        worst_std = np.std(overall_attrs_aucs[2][:, 0])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[2][:, 0]
    else:
        worst_mean = np.mean(overall_attrs_aucs[2][:, 1])
        worst_std = np.std(overall_attrs_aucs[2][:, 1])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[2][:, 1]
    print('Ethnicity: ', round(np.mean(overall_dpds[:, 2]) * 100, 2), ' ± ', round(np.std(overall_dpds[:, 2]) * 100, 2), ' & ', round(np.mean(overall_eods[:, 2]) * 100, 2), ' ± ', round(np.std(overall_eods[:, 2]) * 100, 2), ' & ',
                         round(np.mean(overall_aucs) * 100, 2), ' ± ', round(np.std(overall_aucs) * 100, 2), ' & ', round(np.mean(overall_es_aucs[:, 2]) * 100, 2), ' ± ', round(np.std(overall_es_aucs[:, 2]) * 100, 2), ' & ',
                         round(np.mean(overall_attrs_aucs[2][:, 0]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[2][:, 0]) * 100, 2), ' & ', round(np.mean(overall_attrs_aucs[2][:, 1]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[2][:, 1]) * 100, 2), ' & ',
                         round(worst_mean * 100, 2), ' ± ', round(worst_std * 100, 2))

    print('Ethnicity: ', '(', round(np.percentile(bootstrap_dpds[:, 2], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_dpds[:, 2], 97.5) * 100, 2), ')', ' & ', 
                         '(', round(np.percentile(bootstrap_eods[:, 2], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_eods[:, 2], 97.5) * 100, 2), ')', ' & ',
                         '(', round(np.percentile(bootstrap_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_aucs, 97.5) * 100, 2), ')', ' & ', 
                         '(', round(np.percentile(bootstrap_es_aucs[:, 2], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_es_aucs[:, 2], 97.5) * 100, 2), ')', ' & ',
                         '(', round(np.percentile(bootstrap_attrs_aucs[2][:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[2][:, 0], 97.5) * 100, 2), ')', ' & ', 
                         '(', round(np.percentile(bootstrap_attrs_aucs[2][:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[2][:, 1], 97.5) * 100, 2), ')', ' & ',
                         '(', round(np.percentile(bootstrap_worst_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_worst_aucs, 97.5) * 100, 2), ')')
    
    if np.mean(overall_attrs_aucs[3][:, 0]) < np.mean(overall_attrs_aucs[3][:, 1]) and np.mean(overall_attrs_aucs[3][:, 0]) < np.mean(overall_attrs_aucs[3][:, 2]):
        worst_mean = np.mean(overall_attrs_aucs[3][:, 0])
        worst_std = np.std(overall_attrs_aucs[3][:, 0])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[3][:, 0]
    elif np.mean(overall_attrs_aucs[3][:, 1]) < np.mean(overall_attrs_aucs[3][:, 0]) and np.mean(overall_attrs_aucs[3][:, 1]) < np.mean(overall_attrs_aucs[3][:, 2]):
        worst_mean = np.mean(overall_attrs_aucs[3][:, 1])
        worst_std = np.std(overall_attrs_aucs[3][:, 1])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[3][:, 1]
    else:
        worst_mean = np.mean(overall_attrs_aucs[3][:, 2])
        worst_std = np.std(overall_attrs_aucs[3][:, 2])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[3][:, 2]
    print('Language: ', round(np.mean(overall_dpds[:, 3]) * 100, 2), ' ± ', round(np.std(overall_dpds[:, 3]) * 100, 2), ' & ', round(np.mean(overall_eods[:, 3]) * 100, 2), ' ± ', round(np.std(overall_eods[:, 3]) * 100, 2), ' & ',
                        round(np.mean(overall_aucs) * 100, 2), ' ± ', round(np.std(overall_aucs) * 100, 2), ' & ', round(np.mean(overall_es_aucs[:, 3]) * 100, 2), ' ± ', round(np.std(overall_es_aucs[:, 3]) * 100, 2), ' & ',
                        round(np.mean(overall_attrs_aucs[3][:, 0]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[3][:, 0]) * 100, 2), ' & ', round(np.mean(overall_attrs_aucs[3][:, 1]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[3][:, 1]) * 100, 2), ' & ',
                        round(np.mean(overall_attrs_aucs[3][:, 2]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[3][:, 2]) * 100, 2), ' & ', round(worst_mean * 100, 2), ' ± ', round(worst_std * 100, 2))

    print('Language: ', '(', round(np.percentile(bootstrap_dpds[:, 3], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_dpds[:, 3], 97.5) * 100, 2), ')', ' & ', 
                        '(', round(np.percentile(bootstrap_eods[:, 3], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_eods[:, 3], 97.5) * 100, 2), ')', ' & ',
                        '(', round(np.percentile(bootstrap_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_aucs, 97.5) * 100, 2), ')', ' & ', 
                        '(', round(np.percentile(bootstrap_es_aucs[:, 3], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_es_aucs[:, 3], 97.5) * 100, 2), ')', ' & ',
                        '(', round(np.percentile(bootstrap_attrs_aucs[3][:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[3][:, 0], 97.5) * 100, 2), ')', ' & ', 
                        '(', round(np.percentile(bootstrap_attrs_aucs[3][:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[3][:, 1], 97.5) * 100, 2), ')', ' & ',
                        '(', round(np.percentile(bootstrap_attrs_aucs[3][:, 2], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[3][:, 2], 97.5) * 100, 2), ')', ' & ', 
                        '(', round(np.percentile(bootstrap_worst_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_worst_aucs, 97.5) * 100, 2), ')')

def analyze_fairfundus_npz_results_files(files, n_bootstrap = 1000):
    overall_aucs = []
    overall_es_aucs = []
    overall_aucs_by_attrs = []
    overall_dpds = []
    overall_eods = []
    
    bootstrap_aucs = []
    bootstrap_es_aucs = []
    bootstrap_aucs_by_attrs = []
    bootstrap_dpds = []
    bootstrap_eods = []

    for file in files:
        data = np.load(file)
        preds = data['val_pred']
        gts = data['val_gt']
        attrs = data['val_attr']

        overall_acc, eval_es_acc, overall_auc, eval_es_auc, eval_aucs_by_attrs, eval_dpds, eval_eods, between_group_disparity = evalute_comprehensive_perf(preds, gts, attrs.T)
        overall_aucs.append(overall_auc)
        overall_es_aucs.append(eval_es_auc)
        overall_aucs_by_attrs.append(eval_aucs_by_attrs)
        overall_dpds.append(eval_dpds)
        overall_eods.append(eval_eods)

        strata = np.array([f"{y}_{g}_{a}" for y, g, a in zip(gts, attrs[:, 0], attrs[:, 1])])
        
        for _ in range(n_bootstrap):
            boot_gts, boot_preds, boot_attrs = resample(
                gts, preds, attrs,
                replace=True,
                n_samples=len(gts),
                stratify=strata
            )
            bootstrap_acc, bootstrap_eval_es_acc, bootstrap_auc, bootstrap_eval_es_auc, bootstrap_eval_aucs_by_attrs, bootstrap_eval_dpds, bootstrap_eval_eods, bootstrap_between_group_disparity = evalute_comprehensive_perf(boot_preds, boot_gts, boot_attrs.T)
            bootstrap_aucs.append(bootstrap_auc)
            bootstrap_es_aucs.append(bootstrap_eval_es_auc)
            bootstrap_aucs_by_attrs.append(bootstrap_eval_aucs_by_attrs)
            bootstrap_dpds.append(bootstrap_eval_dpds)
            bootstrap_eods.append(bootstrap_eval_eods)

    overall_aucs = np.array(overall_aucs)
    overall_es_aucs = np.array(overall_es_aucs)
    overall_dpds = np.array(overall_dpds)
    overall_eods = np.array(overall_eods)

    overall_attrs_aucs = [[], []]
    for i in range(2):
        for attr_aucs in overall_aucs_by_attrs:
            overall_attrs_aucs[i].append(attr_aucs[i])
        overall_attrs_aucs[i] = np.array(overall_attrs_aucs[i])
    
    bootstrap_aucs = np.array(bootstrap_aucs)
    bootstrap_es_aucs = np.array(bootstrap_es_aucs)
    bootstrap_dpds = np.array(bootstrap_dpds)
    bootstrap_eods = np.array(bootstrap_eods)

    bootstrap_attrs_aucs = [[], []]
    for i in range(2):
        for attr_aucs in bootstrap_aucs_by_attrs:
            bootstrap_attrs_aucs[i].append(attr_aucs[i])
        bootstrap_attrs_aucs[i] = np.array(bootstrap_attrs_aucs[i])
    
    if np.mean(overall_attrs_aucs[0][:, 0]) < np.mean(overall_attrs_aucs[0][:, 1]):
        worst_mean = np.mean(overall_attrs_aucs[0][:, 0])
        worst_std = np.std(overall_attrs_aucs[0][:, 0])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[0][:, 0]
    else:
        worst_mean = np.mean(overall_attrs_aucs[0][:, 1])
        worst_std = np.std(overall_attrs_aucs[0][:, 1])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[0][:, 1]
    print('Gender: ', round(np.mean(overall_dpds[:, 0]) * 100, 2), ' ± ', round(np.std(overall_dpds[:, 0]) * 100, 2), ' & ', round(np.mean(overall_eods[:, 0]) * 100, 2), ' ± ', round(np.std(overall_eods[:, 0]) * 100, 2), ' & ',
                      round(np.mean(overall_aucs) * 100, 2), ' ± ', round(np.std(overall_aucs) * 100, 2), ' & ', round(np.mean(overall_es_aucs[:, 0]) * 100, 2), ' ± ', round(np.std(overall_es_aucs[:, 0]) * 100, 2), ' & ',
                      round(np.mean(overall_attrs_aucs[0][:, 0]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[0][:, 0]) * 100, 2), ' & ', round(np.mean(overall_attrs_aucs[0][:, 1]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[0][:, 1]) * 100, 2), ' & ',
                      round(worst_mean * 100, 2), ' ± ', round(worst_std * 100, 2))

    print('Gender: ', '(', round(np.percentile(bootstrap_dpds[:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_dpds[:, 0], 97.5) * 100, 2), ')', ' & ', 
                      '(', round(np.percentile(bootstrap_eods[:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_eods[:, 0], 97.5) * 100, 2), ')', ' & ',
                      '(', round(np.percentile(bootstrap_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_aucs, 97.5) * 100, 2), ')', ' & ', 
                      '(', round(np.percentile(bootstrap_es_aucs[:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_es_aucs[:, 0], 97.5) * 100, 2), ')', ' & ',
                      '(', round(np.percentile(bootstrap_attrs_aucs[0][:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[0][:, 0], 97.5) * 100, 2), ')', ' & ', 
                      '(', round(np.percentile(bootstrap_attrs_aucs[0][:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[0][:, 1], 97.5) * 100, 2), ')', ' & ',
                      '(', round(np.percentile(bootstrap_worst_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_worst_aucs, 97.5) * 100, 2), ')')
    
    if np.mean(overall_attrs_aucs[1][:, 0]) < np.mean(overall_attrs_aucs[1][:, 1]) and np.mean(overall_attrs_aucs[1][:, 0]) < np.mean(overall_attrs_aucs[1][:, 2]):
        worst_mean = np.mean(overall_attrs_aucs[1][:, 0])
        worst_std = np.std(overall_attrs_aucs[1][:, 0])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[1][:, 0]
    elif np.mean(overall_attrs_aucs[1][:, 1]) < np.mean(overall_attrs_aucs[1][:, 0]) and np.mean(overall_attrs_aucs[1][:, 1]) < np.mean(overall_attrs_aucs[1][:, 2]):
        worst_mean = np.mean(overall_attrs_aucs[1][:, 1])
        worst_std = np.std(overall_attrs_aucs[1][:, 1])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[1][:, 1]
    else:
        worst_mean = np.mean(overall_attrs_aucs[1][:, 2])
        worst_std = np.std(overall_attrs_aucs[1][:, 2])
        bootstrap_worst_aucs = bootstrap_attrs_aucs[1][:, 2]
    print('Age: ', round(np.mean(overall_dpds[:, 1]) * 100, 2), ' ± ', round(np.std(overall_dpds[:, 1]) * 100, 2), ' & ', round(np.mean(overall_eods[:, 1]) * 100, 2), ' ± ', round(np.std(overall_eods[:, 1]) * 100, 2), ' & ',
                   round(np.mean(overall_aucs) * 100, 2), ' ± ', round(np.std(overall_aucs) * 100, 2), ' & ', round(np.mean(overall_es_aucs[:, 1]) * 100, 2), ' ± ', round(np.std(overall_es_aucs[:, 1]) * 100, 2), ' & ',
                   round(np.mean(overall_attrs_aucs[1][:, 0]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[1][:, 0]) * 100, 2), ' & ', round(np.mean(overall_attrs_aucs[1][:, 1]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[1][:, 1]) * 100, 2), ' & ',
                   round(np.mean(overall_attrs_aucs[1][:, 2]) * 100, 2), ' ± ', round(np.std(overall_attrs_aucs[1][:, 2]) * 100, 2), ' & ', round(worst_mean * 100, 2), ' ± ', round(worst_std * 100, 2))

    print('Age: ', '(', round(np.percentile(bootstrap_dpds[:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_dpds[:, 1], 97.5) * 100, 2), ')', ' & ', 
                   '(', round(np.percentile(bootstrap_eods[:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_eods[:, 1], 97.5) * 100, 2), ')', ' & ',
                   '(', round(np.percentile(bootstrap_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_aucs, 97.5) * 100, 2), ')', ' & ', 
                   '(', round(np.percentile(bootstrap_es_aucs[:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_es_aucs[:, 1], 97.5) * 100, 2), ')', ' & ',
                   '(', round(np.percentile(bootstrap_attrs_aucs[1][:, 0], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[1][:, 0], 97.5) * 100, 2), ')', ' & ', 
                   '(', round(np.percentile(bootstrap_attrs_aucs[1][:, 1], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[1][:, 1], 97.5) * 100, 2), ')', ' & ',
                   '(', round(np.percentile(bootstrap_attrs_aucs[1][:, 2], 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_attrs_aucs[1][:, 2], 97.5) * 100, 2), ')', ' & ', 
                   '(', round(np.percentile(bootstrap_worst_aucs, 2.5) * 100, 2), ' - ', round(np.percentile(bootstrap_worst_aucs, 97.5) * 100, 2), ')')


def print_zero_shot_exp_results(base_folder, sorted_exps, exp_index, n_bootstrap = 1000):
    run_1 = os.path.join(base_folder, sorted_exps[3 * exp_index], 'test')
    run_2 = os.path.join(base_folder, sorted_exps[3 * exp_index + 1], 'test')
    run_3 = os.path.join(base_folder, sorted_exps[3 * exp_index + 2], 'test')

    run1_subfolder = [d for d in os.listdir(run_1) if os.path.isdir(os.path.join(run_1, d))][0]
    run2_subfolder = [d for d in os.listdir(run_2) if os.path.isdir(os.path.join(run_2, d))][0]
    run3_subfolder = [d for d in os.listdir(run_3) if os.path.isdir(os.path.join(run_3, d))][0]

    run_1 = os.path.join(run_1, run1_subfolder, 'pred_gt_ep000.npz')
    run_2 = os.path.join(run_2, run2_subfolder, 'pred_gt_ep000.npz')
    run_3 = os.path.join(run_3, run3_subfolder, 'pred_gt_ep000.npz')

    print(run_1)
    print(run_2)
    print(run_3)
    
    exp_npz_files = [run_1, run_2, run_3]
    
    return analyze_npz_results_files(exp_npz_files, n_bootstrap = n_bootstrap)

def print_linear_probing_exp_results(base_folder, sorted_exps, exp_index, n_bootstrap = 1000):
    run_1 = os.path.join(base_folder, sorted_exps[3 * exp_index], 'test_linprobe_blr_0.00005_layers_4')
    run_2 = os.path.join(base_folder, sorted_exps[3 * exp_index + 1], 'test_linprobe_blr_0.00005_layers_4')
    run_3 = os.path.join(base_folder, sorted_exps[3 * exp_index + 2], 'test_linprobe_blr_0.00005_layers_4')

    run1_subfolder = [d for d in os.listdir(run_1) if os.path.isdir(os.path.join(run_1, d))][0]
    run2_subfolder = [d for d in os.listdir(run_2) if os.path.isdir(os.path.join(run_2, d))][0]
    run3_subfolder = [d for d in os.listdir(run_3) if os.path.isdir(os.path.join(run_3, d))][0]

    run_1 = os.path.join(run_1, run1_subfolder, 'pred_gt_ep000.npz')
    run_2 = os.path.join(run_2, run2_subfolder, 'pred_gt_ep000.npz')
    run_3 = os.path.join(run_3, run3_subfolder, 'pred_gt_ep000.npz')

    print(run_1)
    print(run_2)
    print(run_3)
    
    exp_npz_files = [run_1, run_2, run_3]
    
    return analyze_npz_results_files(exp_npz_files, n_bootstrap = n_bootstrap)

def print_fairfundus_exp_results(base_folder, sorted_exps, exp_index, n_bootstrap = 1000):
    run_1 = os.path.join(base_folder, sorted_exps[3 * exp_index], 'test_linprobe_fair_masc_blr_0.0025_layers_4_5_fold')
    run_2 = os.path.join(base_folder, sorted_exps[3 * exp_index + 1], 'test_linprobe_fair_masc_blr_0.0025_layers_4_5_fold')
    run_3 = os.path.join(base_folder, sorted_exps[3 * exp_index + 2], 'test_linprobe_fair_masc_blr_0.0025_layers_4_5_fold')

    runs_folders = [run_1, run_2, run_3]
    runs_splits_folders = []
    for run_folder in runs_folders:
        for split in range(1, 6):
            runs_splits_folders.append(os.path.join(run_folder, f'split{split}'))

    exp_npz_files = []
    for run_split_folder in runs_splits_folders:
        run_subfolder = [d for d in os.listdir(run_split_folder) if os.path.isdir(os.path.join(run_split_folder, d))][0]
        exp_npz_files.append(os.path.join(run_split_folder, run_subfolder, 'pred_gt_ep000.npz'))

    for file in exp_npz_files:
        print(file)
    
    return analyze_fairfundus_npz_results_files(exp_npz_files, n_bootstrap = n_bootstrap)

if __name__ == '__main__':
    args = parser.parse_args()

    print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed
    set_random_seed(seed)

    sorted_exps = sorted(os.listdir(args.base_folder))

    if args.exp_type == 'zero_shot':
        print_zero_shot_exp_results(args.base_folder, sorted_exps, args.exp_index, n_bootstrap = args.n_bootstrap)
    elif args.exp_type == 'linear_probing':
        print_linear_probing_exp_results(args.base_folder, sorted_exps, args.exp_index, n_bootstrap = args.n_bootstrap)
    elif args.exp_type == 'fairfundus':
        print_fairfundus_exp_results(args.base_folder, sorted_exps, args.exp_index, n_bootstrap = args.n_bootstrap)
