Heart Disease Prediction Using Deep Learning
This project implements deep learning models using Keras to predict heart disease based on medical data.

Overview
Two models are created:

Categorical Model: Predicts heart disease (0: No, 1: Yes).
Binary Model: Predicts probability of heart disease (0 or 1).
Dataset
The dataset includes features like age, sex, blood pressure, cholesterol, etc., and the target indicates heart disease presence. Data preprocessing includes handling missing values and standardizing the dataset.

Model Architecture
Both models use dense layers with dropout regularization:

Input: 13 medical features
Hidden Layers: ReLU activation
Output: Softmax (categorical) / Sigmoid (binary)
Usage
Clone the repo and install dependencies:

bash
Copy code
git clone https://github.com/yourusername/heart-disease-prediction.git
pip install -r requirements.txt
Train and evaluate the models:

python
Copy code
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=10)
Visualize results:

python
Copy code
plt.plot(history.history['accuracy'])
Results
Categorical Model Accuracy: 83.6%
Binary Model: Results from binary classification report.
