import os
import numpy as np
from src.model import load_trained_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Paths to test data and trained model
X_TEST_PATH = "dataset/X_test.npy"
Y_TEST_PATH = "dataset/y_test.npy"
MODEL_PATH = "models/AktionFocusNetz.h5"


def check_files_exist(*files):
    """
    Checks if the specified files exist. Raises a FileNotFoundError if any file is missing.

    Args:
        *files: Variable length argument list of file paths to check.

    Raises:
        FileNotFoundError: If any of the files do not exist.
    """
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File '{file}' not found. Please ensure it exists in the specified path.")


# Check if test data and model file exist
check_files_exist(X_TEST_PATH, Y_TEST_PATH, MODEL_PATH)

# Load the test data
print("Loading the data for testing.")
X_test = np.load(X_TEST_PATH)  # Features for testing
y_test = np.load(Y_TEST_PATH)  # True labels for testing
print("Testing data loaded.")

# Load the trained model
print("Loading the model for testing.")
model = load_trained_model(MODEL_PATH)
print("Model loaded for testing.")

# Make predictions on the test data
print("Testing started.")
y_pred = model.predict(X_test)  # Predict class probabilities for each test sample
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
print("Testing completed.")

# Convert true labels from one-hot encoding to 1-D class labels
y_test_classes = np.argmax(y_test, axis=1)

# Calculate and display test accuracy
accuracy = np.mean(y_pred_classes == y_test_classes)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Generate and print a classification report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Compute and plot the confusion matrix for further evaluation
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()