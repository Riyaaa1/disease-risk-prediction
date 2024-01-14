import pickle 
import streamlit as st
import pandas as pd
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from sklearn.preprocessing import StandardScaler
import numpy as np

with open('/home/riya/wd/model/model.pkl','rb') as model_pickle_file:
    REGRESSOR_MODEL = pickle.load(model_pickle_file)
with open("/home/riya/wd/model/scaler.pkl", "rb") as scaler_file:
    SCALER = pickle.load(scaler_file)

lambda_val=0.21141146338311537

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


def prediction(age:int,
               sbp:int,
               hdl:int,
               tcl:int,
               diabetes:str,
               sex:str
               ):
   cholesterol_ratio=tcl/hdl
   input_data=pd.DataFrame(
      {
      "Age":[age],
      "SBP(mmHg)":[sbp],
      "Cholesterol_ratio":[cholesterol_ratio],
      "HDL":[hdl],
      "TCL":[tcl],
      "Diabetes_y":[diabetes],
      "Sex_m":[sex]}
   )

   scaled_data=SCALER.transform(input_data)
   predictions=REGRESSOR_MODEL.predict(scaled_data)


   return predictions[0]

def categorize_risk_level(score):
    if score > 0.7:
        return "High Risk"
    elif 0.3 < score < 0.7:
        return "Moderate Risk"
    else:
        return "Low Risk"

      

def main():
    st.title("Heart Risk Prediction App")
    # display the data
    st.subheader("Have a look at our dataset")
    data= cleaned_data()
    st.dataframe(data.head())
    st.sidebar.title('Please fill your informations')

    age=st.sidebar.slider("What's your age?",data['Age'].min(),data['Age'].max())
    sex=st.sidebar.selectbox("Select your gender",("Male","Female"))
    sbp=st.sidebar.slider("Value of systolic blood pressure (mmHg)",data["SBP(mmHg)"].min(),data['SBP(mmHg)'].max())
    hdl=st.sidebar.slider("HDL cholesterol (mg/dl)",data['HDL'].min(),data['HDL'].max())
    tcl=st.sidebar.slider("Total cholesterol (mg/dl)",data['TCL'].min(),data['TCL'].max())
    diabetes=st.sidebar.selectbox("Do you have diabetes?",('Yes','No'))
    diabetes_value = 1 if diabetes == "Yes" else 0
    sex_value = 1 if sex == "Male" else 0
    # Predict risk score
    risk_score_transformed = prediction(age,sbp,hdl,tcl,diabetes_value,sex_value)

    risk_score=inv_boxcox(risk_score_transformed,lambda_val)
    # Categorize risk level based on thresholds
    risk_level = categorize_risk_level(risk_score)

    st.subheader("Risk Score Prediction")
    st.write(f"The predicted risk score is: {risk_score:.2f}")
    st.write(f"Risk Level: {risk_level}")
    

if __name__=="__main__":
   main()