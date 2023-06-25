# Breast Cancer Diagnosis Classification

This project involves building a machine learning model to classify breast cancer diagnosis based on various features. It includes data preprocessing, exploratory data analysis, feature scaling, and training multiple classification models. Additionally, it implements a deep learning model using Keras to classify the diagnosis.

## Project Structure
The project consists of the following files:

- <b> Notebook/Notebook.ipynb </b>: Python script containing the code for breast cancer diagnosis classification.
- <b> data/data.csv </b>: Dataset file containing breast cancer diagnosis and feature information.
- <b> requirements.txt </b>: Text file containing the list of required Python packages to run the project.

## Requirements
The project requires the following packages to be installed:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras

To install these packages, you can use the following command:

```bash
pip install -r requirements.txt
```

## Usage
To use this project, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/jaywyawhare/Breast-Cancer-Diagnosis-Classification.git
    ```

1. Navigate to the project directory:

    ```bash
    cd Breast-Cancer-Diagnosis-Classification
    ```

1. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Tasks


The provided code performs the following tasks:

- Preprocesses the breast cancer diagnosis dataset by removing unnecessary columns and replacing zero values.
- Performs exploratory data analysis by plotting histograms of numerical columns.
- Splits the dataset into train, test, and validation sets.
- Encodes categorical columns using label encoding.
- Scales the numerical features using standard scaling.
- Trains various classification models and evaluates their performance using accuracy, precision, recall, F1 score, and confusion matrix.
- Builds and trains a deep learning model using Keras, with a binary cross-entropy loss function and Adam optimizer.
- Plots the training and validation loss curves.

## Conclusion

The provided code demonstrates the process of building a machine learning model for breast cancer diagnosis classification. It shows how to preprocess the data, train multiple classifiers, and evaluate their performance. It also includes an example of building a deep learning model using Keras.

Feel free to modify the code or experiment with different models and techniques based on your specific requirements.

If you have any questions or need further assistance, please feel free to reach out.

*This project is a sample classification task for breast cancer diagnosis and serves as a starting point for further analysis and exploration.*