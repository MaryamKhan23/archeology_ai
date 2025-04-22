from utils.preprocessing import load_images_from_folder
from models.cnn_model import create_cnn
from models.random_forest import create_rf_model
from utils.fault_simulation import add_noise, blur_image, occlude
from majority_vote import majority_voting
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load data
X, y = load_images_from_folder("data/")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 2: Train models
cnn_model = create_cnn()
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

rf_model = create_rf_model(X_train.reshape(len(X_train), -1), y_train)

# Step 3: Make predictions
cnn_preds = (cnn_model.predict(X_test) > 0.5).astype(int)
rf_preds = rf_model.predict(X_test.reshape(len(X_test), -1))

# Step 4: Apply majority voting
final_preds = majority_voting(cnn_preds.flatten(), rf_preds)

# Step 5: Evaluate results
print("Evaluation Report:")
print(classification_report(y_test, final_preds))

# Visualize a result
import matplotlib.pyplot as plt
plt.imshow(X_test[0])
plt.title(f"Prediction: {'Yes' if final_preds[0]==1 else 'No'}")
plt.show()
