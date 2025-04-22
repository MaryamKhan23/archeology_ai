# models/svm_model.py
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def create_svm_model(X_train, y_train):
    """
    Create and train an SVM model for image classification.
    
    Args:
        X_train: Training data (flattened images)
        y_train: Training labels
        
    Returns:
        Trained SVM model
    """
    # Scale features for better SVM performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create SVM classifier with RBF kernel
    model = svm.SVC(
        kernel='rbf',       # Radial basis function kernel works well for image data
        C=10.0,             # Regularization parameter
        gamma='scale',      # Kernel coefficient
        probability=True    # Enable probability estimates
    )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Return both the model and the scaler
    return model, scaler