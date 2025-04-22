import numpy as np
from scipy.stats import mode

def confidence_weighted_voting(*model_preds_with_conf):
    # Weight predictions by confidence scores
    preds, confs = zip(*model_preds_with_conf)
    weighted_preds = np.average(preds, weights=confs, axis=1)
    final_preds = (weighted_preds > 0.5).astype(int)
    return final_preds, weighted_preds