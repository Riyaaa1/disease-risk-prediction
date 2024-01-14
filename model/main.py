import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette(sns.color_palette("Paired"))
import warnings
# Suppress FutureWarnings from Seaborn
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from scipy.stats import boxcox

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pickle

# changing the column names 
def rem_space(name):
    new_name=[]
    for i in name:
        i=i.replace(" ","_")
        new_name.append(i)
    return new_name

# creating a function to detect outlier in the data
def outlier_detection(dataframe):
  Q1 = dataframe.quantile(0.25)
  Q3 = dataframe.quantile(0.75)
  IQR = Q3 - Q1
  upper_end = Q3 + 1.5 * IQR
  lower_end = Q1 - 1.5 * IQR 
  outlier = dataframe[(dataframe > upper_end) | (dataframe < lower_end)]
  return outlier

def create_model(data):
    X=data[['Age',"SBP(mmHg)","Cholesterol_ratio","HDL","TCL","Diabetes_y","Sex_m"]]
    Y=data['Risk_Score']
    x_train,x_test,y_train,y_test=train_test_split(X,
                                               Y,
                                               test_size=0.3,
                                               random_state=42)
    # scaling
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(x_train)
    X_test_scaled=scaler.transform(x_test)
    model=LinearRegression()
    # fit the model
    model.fit(X_train_scaled,y_train)
    print('Intercept:',model.intercept_)        
    print('Coefficients:',model.coef_) 
    # predictions
    y_pred=model.predict(X_test_scaled)

    # evaluation metrics
    r2=r2_score(y_test,y_pred)
    mae = mean_absolute_error(y_pred,y_test) # mae measures how far the predictions are from the actual value, so lower the mae the better
    mse = mean_squared_error(y_pred,y_test)
    rmse=np.sqrt(mean_absolute_error(y_pred,y_test))
    print(f"Accuracy of model:{r2*100:.2f}%\nmean absolute score:{mae:.2f}\nmean squared error:{mse:.2f}\nroot mean squared error:{rmse:.2f}")
    return model,scaler

def cleaned_data():
    data=pd.read_csv("/home/riya/wd/dataset_savioroftheheart.csv")
    new_colnames = rem_space(data.columns)
    data.rename(columns=dict(zip(data.columns,new_colnames)),inplace=True)
    # performing boxcox transformation
    data['Risk_Score']=boxcox(data['Risk_Score'])[0]
    data.drop(data.columns[-1],axis=1,inplace=True)
    SBPoutliers = outlier_detection(data['SBP(mmHg)'])
    data.drop(SBPoutliers.index,inplace=True)
    data['Cholesterol_ratio']=data['TCL']/data['HDL']
    df=pd.get_dummies(data,drop_first=True,dtype=int)
    return df

def main():
    data=cleaned_data()
    model,scaler =create_model(data)

    # exporting the model and scaler as binary files:
    with open("model.pkl","wb") as f:
        pickle.dump(model,f)
    with open("scaler.pkl","wb") as f:
        pickle.dump(scaler,f)

if __name__ == "__main__":
    main()
