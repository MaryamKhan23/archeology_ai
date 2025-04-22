from utils.preprocessing import load_images_from_folder
from models.cnn_model import create_cnn
from models.random_forest import create_rf_model
from models.svm import create_svm_model
from models.gradient_boosting import create_gb_model
from utils.fault_simulation import add_noise, blur_image, occlude
from majority_vote import majority_voting, confidence_weighted_voting
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

print("Starting archaeological site detection system...")

# Step 1: Load data
print("Loading and preprocessing images...")
try:
    X, y = load_images_from_folder("data/")
    print(f"Loaded {len(X)} images with {sum(y)} positive examples and {len(y) - sum(y)} negative examples")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# Step 2: Train models
print("Training machine learning models...")

# Train CNN model
print("Training CNN model...")
cnn_model = create_cnn(input_shape=X_train[0].shape)
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# Flatten images for traditional ML models
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

# Train Random Forest model
print("Training Random Forest model...")
rf_model = create_rf_model(X_train_flat, y_train)

# Train SVM model
print("Training SVM model...")
svm_model, scaler = create_svm_model(X_train_flat, y_train)

# Train Gradient Boosting model
print("Training Gradient Boosting model...")
gb_model = create_gb_model(X_train_flat, y_train)

# Step 3: Make predictions
print("Making predictions with each model...")
cnn_preds = (cnn_model.predict(X_test) > 0.5).astype(int).flatten()
rf_preds = rf_model.predict(X_test_flat)
X_test_scaled = scaler.transform(X_test_flat)
svm_preds = svm_model.predict(X_test_scaled)
gb_preds = gb_model.predict(X_test_flat)

# Get confidence scores (probabilities) where available
cnn_conf = cnn_model.predict(X_test).flatten()
rf_conf = rf_model.predict_proba(X_test_flat)[:, 1] if hasattr(rf_model, 'predict_proba') else np.ones_like(rf_preds)
svm_conf = svm_model.predict_proba(X_test_scaled)[:, 1]
gb_conf = gb_model.predict_proba(X_test_flat)[:, 1]

# Step 4: Apply majority voting
print("Applying majority voting strategy...")
simple_majority_preds = majority_voting(cnn_preds, rf_preds, svm_preds, gb_preds)

# Also try confidence-weighted voting
confidence_preds, confidence_scores = confidence_weighted_voting(
    (cnn_preds, cnn_conf),
    (rf_preds, rf_conf),
    (svm_preds, svm_conf),
    (gb_preds, gb_conf)
)

# Step 5: Evaluate results
print("\n--- EVALUATION RESULTS ---")
print("\nSimple Majority Voting:")
print(classification_report(y_test, simple_majority_preds))
print("\nConfidence-Weighted Voting:")
print(classification_report(y_test, confidence_preds))

# Display confusion matrices
print("\nConfusion Matrix (Simple Majority):")
print(confusion_matrix(y_test, simple_majority_preds))
print("\nConfusion Matrix (Confidence-Weighted):")
print(confusion_matrix(y_test, confidence_preds))

# Step 6: Test fault tolerance
print("\n--- FAULT TOLERANCE TESTING ---")

# Create faulty test sets
print("Testing system with simulated faults...")
X_test_noisy = np.array([add_noise(img) for img in X_test])
X_test_blurry = np.array([blur_image(img) for img in X_test])
X_test_occluded = np.array([occlude(img.copy()) for img in X_test])

# Make predictions on faulty data
noisy_preds = majority_voting(
    (cnn_model.predict(X_test_noisy) > 0.5).astype(int).flatten(),
    rf_model.predict(X_test_noisy.reshape(len(X_test_noisy), -1)),
    svm_model.predict(scaler.transform(X_test_noisy.reshape(len(X_test_noisy), -1))),
    gb_model.predict(X_test_noisy.reshape(len(X_test_noisy), -1))
)

blurry_preds = majority_voting(
    (cnn_model.predict(X_test_blurry) > 0.5).astype(int).flatten(),
    rf_model.predict(X_test_blurry.reshape(len(X_test_blurry), -1)),
    svm_model.predict(scaler.transform(X_test_blurry.reshape(len(X_test_blurry), -1))),
    gb_model.predict(X_test_blurry.reshape(len(X_test_blurry), -1))
)

occluded_preds = majority_voting(
    (cnn_model.predict(X_test_occluded) > 0.5).astype(int).flatten(),
    rf_model.predict(X_test_occluded.reshape(len(X_test_occluded), -1)),
    svm_model.predict(scaler.transform(X_test_occluded.reshape(len(X_test_occluded), -1))),
    gb_model.predict(X_test_occluded.reshape(len(X_test_occluded), -1))
)

print("\nPerformance on Noisy Images:")
print(classification_report(y_test, noisy_preds))

print("\nPerformance on Blurry Images:")
print(classification_report(y_test, blurry_preds))

print("\nPerformance on Partially Occluded Images:")
print(classification_report(y_test, occluded_preds))

# Step 7: Visualize results
print("\nSaving result visualizations...")

# Function to visualize predictions
def visualize_predictions(X_images, true_labels, predictions, confidence=None, title_prefix=""):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(6, len(X_images))):
        ax = axes[i]
        ax.imshow(X_images[i])
        
        pred_label = "Archaeological" if predictions[i] == 1 else "Not Archaeological"
        true_label = "Archaeological" if true_labels[i] == 1 else "Not Archaeological"
        
        title = f"{title_prefix}\nPred: {pred_label}, True: {true_label}"
        if confidence is not None:
            title += f"\nConfidence: {confidence[i]:.2f}"
            
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    return fig

# Visualize regular predictions
vis_fig = visualize_predictions(
    X_test, y_test, confidence_preds, confidence_scores, "Archaeological Site Detection"
)
vis_fig.savefig('results/predictions_visualization.png')

# Visualize predictions on faulty data
vis_noisy_fig = visualize_predictions(
    X_test_noisy, y_test, noisy_preds, title_prefix="Noisy Image"
)
vis_noisy_fig.savefig('results/noisy_predictions.png')

vis_blurry_fig = visualize_predictions(
    X_test_blurry, y_test, blurry_preds, title_prefix="Blurry Image"
)
vis_blurry_fig.savefig('results/blurry_predictions.png')

vis_occluded_fig = visualize_predictions(
    X_test_occluded, y_test, occluded_preds, title_prefix="Occluded Image"
)
vis_occluded_fig.savefig('results/occluded_predictions.png')

print("\nProcessing complete! Results saved to 'results/' directory.")