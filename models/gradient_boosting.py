from sklearn.ensemble import GradientBoostingClassifier

def create_gb_model(X_train, y_train):
    """
    Create and train a Gradient Boosting model for image classification.
    
    Args:
        X_train: Training data (flattened images)
        y_train: Training labels
        
    Returns:
        Trained Gradient Boosting model
    """
    # Create Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        n_estimators=100,    # Number of boosting stages
        learning_rate=0.1,   # Shrinks the contribution of each tree
        max_depth=3,         # Maximum depth of the individual trees
        min_samples_split=2, # Minimum samples required to split a node
        subsample=0.8,       # Fraction of samples used for fitting the trees
        random_state=42      # For reproducibility
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model