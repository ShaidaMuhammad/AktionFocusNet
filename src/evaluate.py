import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from src.data_preparation import load_data
import numpy as np
import pandas as pd


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(df_cm, annot=True, cmap="YlGnBu", square=True)
    plt.show()

def evaluate_model(data_path, model_path):
    X_train, X_test, y_train, y_test, label_encoder = load_data(data_path)
    num_classes = len(label_encoder.classes_)
    y_test_categorical = tf.keras.utils.to_categorical(y_test, num_classes)

    model = load_model(model_path)
    y_pred = model.predict(X_test).argmax(axis=1)
    y_test = y_test_categorical.argmax(axis=1)

    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    plot_confusion_matrix(y_test, y_pred, label_encoder.classes_)
