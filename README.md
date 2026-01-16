This project is a machine learning-based stress level prediction system built using Random Forest Regression. 
It predicts a numeric stress score based on user-provided features through an interactive command-line interface.
The model achieves a Mean  Absolute Error(MAE) of 0.3 on a target range of 0.0 to 9.7 indicating strong predictive accuracy.

Stress has a major impact on mental health and productivity. This project aims to:
Predict stress level from input features.
Allow users to interactively enter their own data.
Evaluate model performance using a proper validation approach.
KEY FEATURES:
Interactive user input via terminal
Feature preprocessing for model compatibility.
Random Forest Regressor for prediction.
Train/Test split for model validation.
Performance evaluation using Mean Absolute Error(MAE)
clear output of: (Predicted Stress level, Model MAE)

MODEL & VALIDATION:
Algorithm: Random Forest Regression
Validation Method: Train/Test Split
Target Variable Range: 0.0 - 9.7
Mean Absolute Error (MAE): 0.3
NB/: The low MAE relative to the target range shows that the model generalizes well to unseen data.

TECH STACK:
Python
Pandas
Scikit-learn

HOW IT WORKS:
The dataset is split into Training and testing sets using train_test_split.
A RandomForestRegressor is trained on the TRAINING DATA.
Model performance is Evaluated on test set using MAE.
The user is prompted to input feature values.
The input is transformed and passed to the trained model.
The predicted stress level and MAE are displayed.

HOW TO RUN
git clone https://github.com/christopher796/Stress-Level-Predictor.git
cd Stress-Level-Predictor
python main.py

FUTURE IMPROVEMENTS
Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
Cross-validation for more robust evaluation.
Feature importance visualization.
Web interface using Streamlit or Flask.
Model persistance using joblib or pickle.

AUTHOR:
Christopher Wainaina Ndaru.
Aspiring Machine Learning Engineer interested in practical ML solutions and predictive modeling.
