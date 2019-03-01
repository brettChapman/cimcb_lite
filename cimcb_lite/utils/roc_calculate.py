import numpy as np
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils import resample
from .binary_metrics import binary_metrics


def roc_calculate(Ytrue, Yscore, bootnum=1000, metric=None, val=None):
    """Calculates required metrics for the roc plot function."""
    
    # Get fpr, tpr
    fpr, tpr, threshold = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)
    
    # fpr, tpr with drop_intermediates for fpr = 0 (useful for plot... since we plot specificity on x-axis, we don't need intermediates when fpr=0)
    tpr0 = tpr[fpr == 0][-1]
    tpr = np.concatenate([[tpr0], tpr[fpr > 0]])
    fpr = np.concatenate([[0], fpr[fpr > 0]])
    
    # if metric is provided, calculate stats
    if metric is not None:
        specificity, sensitivity, threshold = get_spec_sens_cuttoff(Ytrue, Yscore, metric, val)
        stats = get_stats(Ytrue, Yscore, specificity)
        stats['val_specificity'] = specificity
        stats['val_sensitivity'] = specificity
        stats['val_cutoffscore'] = threshold 
        
    # bootstrap using vertical averaging
    tpr_boot = []
    boot_stats = []
    for i in range(bootnum):
        # Resample and get tpr, fpr
        Ytrue_res, Yscore_res = resample(Ytrue, Yscore) 
        fpr_res, tpr_res, threshold_res = metrics.roc_curve(Ytrue_res, Yscore_res, pos_label=1, drop_intermediate=False)

        # Drop intermediates when fpr=0
        tpr0_res = tpr_res[fpr_res == 0][-1]
        tpr_res = np.concatenate([[tpr0_res], tpr_res[fpr_res > 0]])
        fpr_res = np.concatenate([[0], fpr_res[fpr_res > 0]])

        # Vertical averaging... use closest fpr_res to fpr, and append the corresponding tpr
        idx = [np.abs(i - fpr_res).argmin() for i in fpr]
        tpr_list = tpr_res[idx]
        tpr_boot.append(tpr_list)
        
        # if metric is provided, calculate stats
        if metric is not None:
            stats_res = get_stats(Ytrue_res, Yscore_res, specificity)
            boot_stats.append(stats_res)
            
    # Get CI for bootstat
    if metric is not None:
        bootci_stats = {}
        for i in boot_stats[0].keys():
            stats_i = [k[i] for k in boot_stats]
            lowci = np.percentile(stats_i,2.5)
            uppci = np.percentile(stats_i, 97.5)
            bootci_stats[i] = [lowci, uppci]
        
    # Get CI for tpr 
    tpr_lowci = np.percentile(tpr_boot, 2.5, axis=0)
    tpr_uppci = np.percentile(tpr_boot, 97.5, axis=0)

    # Add the starting 0 
    tpr = np.insert(tpr, 0, 0)
    fpr = np.insert(fpr, 0, 0)
    tpr_lowci = np.insert(tpr_lowci, 0, 0)
    tpr_uppci = np.insert(tpr_uppci, 0, 0)
    
    # Concatenate tpr_ci
    tpr_ci = np.array([tpr_lowci, tpr_uppci])
    
    if metric is None:
        return fpr, tpr, tpr_ci
    else:
        return fpr, tpr, tpr_ci, stats, bootci_stats
    

def get_sens_spec(Ytrue, Yscore, cuttoff_val):
    """Get sensitivity and specificity from cutoff value."""
    Yscore_round = np.where(np.array(Yscore) > cuttoff_val, 1, 0) 
    tn, fp, fn, tp = metrics.confusion_matrix(Ytrue, Yscore_round).ravel()
    sensitivity  = tp / (tp + fn)
    specificity  = tn / (tn + fp)
    return sensitivity, specificity

def get_sens_cuttoff(Ytrue, Yscore, specificity_val):
    """Get sensitivity and cuttoff value from specificity."""
    fpr0 = 1 - specificity_val
    fpr, sensitivity , thresholds = metrics.roc_curve(Ytrue, Yscore, pos_label=1, drop_intermediate=False)
    idx = np.abs(fpr - fpr0).argmin() # this find the closest value in fpr to fpr0
    return sensitivity[idx], thresholds[idx]

def get_spec_sens_cuttoff(Ytrue, Yscore, metric, val):
    """Return specificity, sensitivity, cutoff value provided the metric and value used."""
    if metric == 'specificity':
        specificity = val
        sensitivity, threshold = get_sens_cuttoff(Ytrue, Yscore, val)
    elif metric == 'cutoffscore':
        threshold = val
        sensitivity, specificity = get_sens_spec(Ytrue, Yscore, val)
    return specificity, sensitivity, threshold

def get_stats(Ytrue, Yscore, specificity):
    """Calculates binary metrics given the specificity."""
    sensitivity, cutoffscore = get_sens_cuttoff(Ytrue, Yscore, specificity)
    stats = binary_metrics(Ytrue, Yscore, cut_off=cutoffscore)
    return stats