# AktionFocusNet: An Attention-Based Neural Network for Human Activity Recognition

This project provides a deep learning model, **AktionFocusNet**, designed to recognize various human activities using inertial sensor data. This code accompanies the research paper **"Paying Attention to Human Activities: Recognizing Daily Activities with an Attention-based Neural Network and Inertial Sensors."**

## Project Structure

- `colab-notebook/`: Contains Jupyter notebooks for training and testing on Google Colab.
  - `AktionFocusNetz_acc_HAR_WISDM11.ipynb`: Notebook for training and evaluating the model on Colab.
- `dataset/`: Holds datasets and preprocessed data files.
  - `onlyMagWISDM11.csv`: Source data used for training.
  - `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`: Preprocessed data for training and testing.
- `models/`: Stores trained models.
  - `AktionFocusNet.h5`: Pretrained model ready for evaluation.
- `src/`: Contains source code files.
  - `data_preparation.py`: Functions for loading and preprocessing data.
  - `model.py`: Defines the neural network model architecture.
  - `train.py`: Script to train the model.
  - `evaluate.py`: Evaluates model performance.
  - `utils.py`: Utility functions to support model training and testing.
- `requirements.txt`: List of required packages.
- `README.md`: Project documentation.

## Setup

### Prerequisites

Ensure you have Python 3.10.12 installed on your machine.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/ShaidaMuhammad/AktionFocusNet.git
    cd AktionFocusNet
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Data

Ensure the following files are available in the `dataset/` directory:

- `onlyMagWISDM11.csv`: Original dataset used for training.
- `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`: Preprocessed datasets for model evaluation.

## Running the Project

### Training

1. Verify that `onlyMagWISDM11.csv` is available in the `dataset/` directory.
2. Run the training script using:

    ```bash
    python train.py
    ```

### Testing

1. Ensure `X_test.npy` and `y_test.npy` are present in the `dataset/` directory.
2. Run the testing script to evaluate the pretrained model:

    ```bash
    python test.py
    ```

### Colab Notebook

You can also use the Jupyter notebook provided in `colab-notebook/AktionFocusNetz_acc_HAR_WISDM11.ipynb` for training and testing on Google Colab.


## Notes

- Ensure that the dataset files are correctly placed in the `dataset/` directory before running the scripts.
- This code was developed and tested on Windows 11 but should be compatible with other operating systems.

## Authors

- **[Shaida Muhammad](https://scholar.google.com/citations?user=XzAuQjMAAAAJ)**
- **Hamza Ali Imran**
- **Kiran Hamza**
- **Saad Wazir**
- **Ataul Aziz Ikram**
