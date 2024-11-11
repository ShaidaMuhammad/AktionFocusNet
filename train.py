import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from src.data_preparation import load_data
from src.model import create_model

def train_model(data_path, epochs=60, batch_size=120):
    """
    Trains a deep learning model on the provided dataset for human activity recognition.

    Args:
        data_path (str): Path to the dataset file (CSV) containing training and test data.
        epochs (int): Number of epochs to train the model. Default is 60.
        batch_size (int): Batch size to use for training. Default is 120.

    Returns:
        model (tf.keras.Model): The trained Keras model.
        label_encoder (LabelEncoder): LabelEncoder object for converting between label classes and integers.
    """
    # Load and preprocess data from CSV file
    X_train, X_test, y_train, y_test, label_encoder = load_data(data_path)
    num_classes = len(label_encoder.classes_)  # Number of unique classes in the dataset
    input_shape = X_train.shape[1:]  # Shape of input data (time periods, features)

    # Convert labels to categorical (one-hot encoding) for model compatibility
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Create the model architecture using the specified input shape and number of classes
    model = create_model(input_shape, num_classes)

    # Train the model with specified hyperparameters and save the training history
    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size, verbose=1
    )

    # Plot training and validation loss over epochs
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy over epochs
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Evaluate the model on the test set and print the accuracy
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print("Test accuracy = {:.2f}%".format(accuracy * 100))

    # Save the trained model to the specified directory
    model.save('models/AktionFocusNetz.h5')
    return model, label_encoder

if __name__ == "__main__":
    # Train the model on the specified dataset with a reduced number of epochs for quick testing
    model, le = train_model('dataset/onlyMagWISDM11.csv', epochs=60)
