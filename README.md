Heart Disease Prediction Using Deep Learning
This repository implements a deep learning model using Keras and TensorFlow to predict the likelihood of heart disease based on patient data. The dataset contains medical information such as age, sex, chest pain type, cholesterol levels, and more, which are used as features for training two models: a categorical model and a binary model.

Table of Contents
Overview
Dataset
Model Architecture
Installation
Usage
Results
Contributing
License
Overview
This project utilizes deep learning models to predict whether a patient is likely to suffer from heart disease. The dataset is preprocessed to handle missing values, standardize the data, and split it into training and test sets. Two models are created:

Categorical Model: Predicts two classes (0: No heart disease, 1: Heart disease).
Binary Model: Predicts the probability of heart disease as either 0 or 1.
The models use dense layers with dropout regularization to prevent overfitting and are evaluated based on accuracy, loss, and precision-recall metrics.

Dataset
The dataset used in this project is the Heart Disease UCI dataset, which includes the following features:

Age
Sex
Chest pain type (cp)
Resting blood pressure (trestbps)
Cholesterol (chol)
Fasting blood sugar (fbs)
Resting ECG results (restecg)
Maximum heart rate achieved (thalach)
Exercise-induced angina (exang)
Oldpeak (depression induced by exercise)
Slope of the peak exercise ST segment (slope)
Number of major vessels colored by fluoroscopy (ca)
Thalassemia (thal)
Target (0: No disease, 1: Disease)
Model Architecture
Categorical Model
Input: 13 features
Dense Layer (16 units, ReLU activation)
Dropout Layer (25% dropout rate)
Dense Layer (8 units, ReLU activation)
Dropout Layer (25% dropout rate)
Output Layer: Softmax activation for 2 classes
Binary Model
Input: 13 features
Dense Layer (16 units, ReLU activation)
Dropout Layer (25% dropout rate)
Dense Layer (8 units, ReLU activation)
Dropout Layer (25% dropout rate)
Output Layer: Sigmoid activation for binary classification
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Download the dataset and place it in the root directory.

Usage
Preprocess the data:

Handle missing values.
Standardize the dataset.
Split data into training and test sets.
Train the categorical model:

python
Copy code
model = create_model()
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=10)
Train the binary model:

python
Copy code
binary_model = create_binary_model()
history = binary_model.fit(X_train, Y_train_binary, validation_data=(X_test, Y_test_binary), epochs=50, batch_size=10)
Evaluate both models using accuracy, loss, and classification reports.

Results
Categorical Model:
Accuracy: 83.6%
F1-Score: 81% for class 0, 86% for class 1
Binary Model:
Accuracy: Results provided after evaluating the binary model using the binary_pred.
Visualization:

python
Copy code
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()
Classification report and accuracy metrics are also provided for both models.

Contributing
If you want to contribute to this project, feel free to open issues or submit pull requests.

License
This project is licensed under the MIT License.
