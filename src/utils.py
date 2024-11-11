import numpy as np


def decode_predictions(y_pred, label_encoder):
    """
    Decodes one-hot encoded model predictions into original class labels.

    Args:
        y_pred (np.array): Array of model predictions in one-hot encoded format (shape: [num_samples, num_classes]).
        label_encoder (LabelEncoder): Fitted LabelEncoder instance used to encode class labels initially.

    Returns:
        np.array: Array of decoded class labels corresponding to the model predictions.
    """
    return label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
