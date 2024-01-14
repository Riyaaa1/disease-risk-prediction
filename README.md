# Machine Learning based cardiovascular risk prediction
## Summary 
In this project, I utilized multivariate linear regression for heart disease risk prediction. I performed necessary pre-processing, EDA, feature engineering, one hot encoding, and model evaluation.

## Approach
I explored the data structure and the contents. Pre-processing steps included different visualisations, outlier detection and removal, transformation of target variable to remove skewness,feature engineering, and one-hot encoding for categorical variables. Selected best features and split the data into train and test sets(70:30). Subsequently, trained a linear regression model on the training data and assessed its performance using metrics such as R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). To validate the model assumptions, visualized the residual plot, ensuring the linearity assumptions of regression were met. Deployed model into an interactive streamlit app.

Link to notebook: [Heart Disease Risk Prediction notebook](https://github.com/Riyaaa1/disease-risk-prediction/blob/main/heart_risk_prediction.ipynb)

Code for app: [Heart Disease Risk Predictor](https://github.com/Riyaaa1/disease-risk-prediction/blob/main/app/app.py)
<br>
<img height="500" src="https://github.com/Riyaaa1/disease-risk-prediction/blob/main/app/Screenshot%20from%202024-01-14%2014-29-38.png">
<br>
