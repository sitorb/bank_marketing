# Bank Marketing Prediction

## Overview

This project aims to predict whether a customer will subscribe to a term deposit based on various factors like demographics, financial status, and previous marketing campaign interactions. A classification model is built using a dataset from a Portuguese bank marketing campaign.

## Dataset

The dataset used is "bank-full.csv", containing information about bank customers. It includes features like age, job, marital status, education, balance, loan, contact, and the target variable 'y' indicating whether the customer subscribed to a term deposit.

## Code Structure

The code is structured into the following steps:

1. **Data Loading and Preprocessing:**
   - Importing necessary libraries (pandas, NumPy, scikit-learn, TensorFlow/Keras).
   - Loading the dataset using pandas.
   - Handling missing values (if any).
   - Converting categorical features to numerical using one-hot encoding.
   - Scaling numerical features using StandardScaler.

2. **Model Building:**
   - Splitting the data into training and testing sets.
   - Creating a neural network model using TensorFlow/Keras with dense layers, dropout for regularization, and an output layer with sigmoid activation.

3. **Model Compilation and Training:**
   - Compiling the model with 'adam' optimizer, 'binary_crossentropy' loss, and 'accuracy' metric.
   - Training the model using the training data with early stopping to prevent overfitting.

4. **Model Evaluation:**
   - Evaluating the model's performance on the test data using metrics like accuracy, precision, recall, and F1-score.

## Logic and Algorithms

- **Classification:** The project uses a classification approach to predict the target variable ('y').
- **Neural Network:** A feedforward neural network with multiple layers is used as the classification model.
- **One-Hot Encoding:** Categorical features are converted to numerical using one-hot encoding, creating new binary columns for each category.
- **Feature Scaling:** Numerical features are scaled to ensure they have similar ranges, improving model performance.
- **Regularization:** Dropout and L2 regularization are used to prevent overfitting and improve generalization.
- **Early Stopping:** Training is stopped early if the validation loss doesn't improve for a certain number of epochs to prevent overfitting.

## Technology Used

- **Python:** The primary programming language used for data processing, model building, and evaluation.
- **Pandas:** Used for data manipulation and analysis.
- **NumPy:** Used for numerical computations.
- **Scikit-learn:** Used for data preprocessing and model evaluation.
- **TensorFlow/Keras:** Used for building and training the neural network model.

![image](https://github.com/user-attachments/assets/78a02d4c-5269-4af8-8d4e-3395a43f8e89)

![image](https://github.com/user-attachments/assets/796893b1-ffdc-4487-8070-62ed509e6bec)


## Conclusion

This project demonstrates the application of machine learning techniques to predict customer behavior in a marketing campaign. The neural network model achieves good performance on the dataset, providing valuable insights for targeted marketing strategies.
