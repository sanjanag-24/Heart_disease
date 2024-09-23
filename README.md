
# Heart Disease Prediction Using Deep Learning

## Overview

This project implements deep learning models using Keras to predict heart disease based on medical data.

## Models

- **Categorical Model**: Predicts heart disease (0: No, 1: Yes).
- **Binary Model**: Predicts heart disease probability (0 or 1).

## Dataset

The dataset includes features like `age`, `sex`, `blood pressure`, `cholesterol`, and more, used to predict heart disease.

## Model Architecture

- **Input**: 13 medical features.
- **Hidden Layers**: ReLU activation with dropout regularization.
- **Output**: Softmax (categorical) / Sigmoid (binary).

## How to Use

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   ```

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Train the models:  
   ```python
   model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=10)
   ```

4. Visualize results:  
   ```python
   plt.plot(history.history['accuracy'])
   ```

## Results

- **Categorical Model Accuracy**: 83.6%
- **Binary Model**: Evaluated using binary classification report.

