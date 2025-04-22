import numpy as np
from scipy.stats import mode

def majority_voting(*predictions):
    """
    Perform simple majority voting among model predictions.
    
    Args:
        *predictions: Variable number of prediction arrays
    
    Returns:
        Array of final predictions based on majority vote
    """
    # Stack predictions horizontally to create a matrix
    stacked_preds = np.column_stack(predictions)
    
    # Use mode to find the most common prediction for each sample
    final_preds, _ = mode(stacked_preds, axis=1)
    
    return final_preds.flatten()

def confidence_weighted_voting(*model_preds_with_conf):
    """
    Weight predictions by confidence scores for final decision.
    
    Args:
        *model_preds_with_conf: Tuples of (predictions, confidence_scores)
    
    Returns:
        Tuple of (final_predictions, confidence_scores)
    """
    # Extract predictions and confidence scores
    preds, confs = zip(*model_preds_with_conf)
    
    # Convert to numpy arrays if they're not already
    preds = [np.array(p) for p in preds]
    confs = [np.array(c) for c in confs]
    
    # Stack predictions and confidences
    stacked_preds = np.stack(preds, axis=1)  # Shape: (n_samples, n_models)
    stacked_confs = np.stack(confs, axis=1)  # Shape: (n_samples, n_models)
    
    # Calculate weighted predictions
    weighted_votes = stacked_preds * stacked_confs
    total_confidence = np.sum(stacked_confs, axis=1, keepdims=True)
    weighted_sum = np.sum(weighted_votes, axis=1) / np.sum(stacked_confs, axis=1)
    
    # Get final predictions based on weighted votes
    final_preds = (weighted_sum > 0.5).astype(int)
    
    return final_preds, weighted_sum