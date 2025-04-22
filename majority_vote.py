import numpy as np
from scipy.stats import mode

def majority_voting(*model_preds):
    all_preds = np.stack(model_preds, axis=1)
    final_preds = mode(all_preds, axis=1)[0]
    return final_preds