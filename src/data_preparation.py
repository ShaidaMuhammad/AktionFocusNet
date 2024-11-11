import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Constants
TIME_PERIODS = 256  # Number of time steps in each segment
STEP_DISTANCE = 16  # Step size for the sliding window approach
LABEL = 'activity'  # Column name for the activity labels in the dataset


def read_data(file_path):
    """
    Reads the dataset from a CSV file and prepares it for processing.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        df (pd.DataFrame): DataFrame with the dataset where 'AccMag_sub9_8' column is cast to float type.
    """
    df = pd.read_csv(file_path)
    df['AccMag_sub9_8'] = df['AccMag_sub9_8'].astype('float')  # Ensure accelerometer data is float
    return df


def encode_labels(df):
    """
    Encodes activity labels as integers.

    Args:
        df (pd.DataFrame): DataFrame containing the dataset with a column for activity labels.

    Returns:
        df (pd.DataFrame): Updated DataFrame with encoded activity labels.
        le (LabelEncoder): Fitted LabelEncoder instance for decoding labels later.
    """
    le = preprocessing.LabelEncoder()  # Initialize Label Encoder
    df[LABEL] = le.fit_transform(df[LABEL].values.ravel())  # Encode activity labels as integers
    return df, le


def create_segments_and_labels(df, time_steps=TIME_PERIODS, step=STEP_DISTANCE):
    """
    Creates segments of accelerometer data and corresponding activity labels.

    Args:
        df (pd.DataFrame): DataFrame containing accelerometer data and encoded labels.
        time_steps (int): Number of time steps for each segment (default is 256).
        step (int): Step size for creating overlapping segments (default is 16).

    Returns:
        reshaped_segments (np.array): Array of shape (num_segments, time_steps, 1) with accelerometer data.
        labels (np.array): Array of shape (num_segments,) containing the majority label for each segment.
    """
    segments = []
    labels = []

    # Iterate through the dataset to create segments with a sliding window
    for i in range(0, len(df) - time_steps, step):
        xs = df['AccMag_sub9_8'].values[i: i + time_steps]  # Segment of accelerometer data
        label = stats.mode(df[LABEL][i: i + time_steps])[0]  # Most common label in the segment
        segments.append([xs])
        labels.append(label)

    # Reshape segments to (num_segments, time_steps, 1) for model input compatibility
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, 1)
    labels = np.asarray(labels)
    return reshaped_segments, labels


def load_data(file_path, test_size=0.3):
    """
    Loads and preprocesses the dataset, then splits it into training and testing sets.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
        test_size (float): Proportion of data to use as the test set (default is 0.3).

    Returns:
        X_train (np.array): Training data of shape (num_train_segments, time_steps, 1).
        X_test (np.array): Testing data of shape (num_test_segments, time_steps, 1).
        y_train (np.array): Encoded labels for training data.
        y_test (np.array): Encoded labels for testing data.
        le (LabelEncoder): LabelEncoder instance for decoding labels.
    """
    # Step 1: Read and preprocess data
    df = read_data(file_path)
    df, le = encode_labels(df)  # Encode labels as integers

    # Step 2: Create segments and labels
    X, y = create_segments_and_labels(df)

    # Step 3: Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test, le