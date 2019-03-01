import numpy as np
import warnings
from sklearn.metrics import confusion_matrix, roc_auc_score


def binary_metrics(Y_true, Y_pred, cut_off=0.5):
    """Return a dict of binary stats with the following metrics: R2, auc, accuracy, precision, sensitivity, specificity, and F1 score."""
    
    Y_pred_round = np.where(np.array(Y_pred) >= cut_off, 1, 0) 
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred_round).ravel() 
    
    # Binary statistics dictionary
    stats = {}
    stats["R2"] = 1 - (sum((Y_true - Y_pred) ** 2) / sum((Y_true - np.mean(Y_true)) ** 2))
    stats["auc"] = roc_auc_score(Y_true, Y_pred)
    stats["accuracy"] = safe_div((tp + tn) , (tp + tn + fp + fn))
    stats["precision"] = safe_div((tp) , (tp + fp))
    stats["sensitivity"] = safe_div((tp) , (tp + fn))
    stats["specificity"] = safe_div((tn) , (tn + fp))
    stats["F1score"] = safe_div((2 * tp) , (2 * tp + fp + fn))

    return stats

def safe_div(x,y):
    if y == 0:
        return np.nan
    return x / y