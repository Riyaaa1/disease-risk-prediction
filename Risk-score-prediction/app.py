import pickle
import streamlit as st
import pandas as pd

# loading the trained model 
# Global variables in bold
model_pickle_file = open('regressor.pkl','rb')
REGRESSOR_MODEL = pickle.load(model_pickle_file)



def prediction(age:int,
               sex:str,
               smoking:str,
               diabetes:str,
               ht_treatment:str,
               hdl:int,
               tcl:int,
               sbp:int
               )->str:
    """This function does the following things:
    1. Changes the categorical values to 1's and 0's.
    2. Sends the user input values to gather the prediction.
    3. 

    Parameters
    ----------
    age : int
        input age.
    sex : str
        input sex.
    smoking : str
        input smoking status.
    diabetes : str
        input diabetes or not.
    ht_treatment : str
        input heart treatment taken or not .
    hdl : int
        input hdl level.
    tcl : int
        input tcl level.
    sbp : int
        input sbp level.

    Returns
    -------
    str
        Returns the prediction
    """
    sex = 1 if sex == 'Male' else 0 
    smoking = 1 if smoking == 'Yes' else 0
    diabetes = 1 if diabetes == 'Yes' else 0
    ht_treatment = 1 if ht_treatment == 'Yes' else 0

    #making predictions
    prediction = REGRESSOR_MODEL.predict(
        [[age,sex,smoking,diabetes,ht_treatment,hdl,tcl,sbp]])
    print(prediction)

    """Since the value that is returned is the prediction the below if else doesnt make sense."""
    if prediction >= 0.5:
        pred = 'High risk'
    else:
        pred = 'Low risk'
    return prediction


def render_prediction():
    """
    Contains streamlit design elements and calls the prediction function after gathering all the input from the user.
    """

    st.header("Streamlit Riskscore prediction ML app")
    # display the data
    st.subheader("Have a look at our dataset")
    data= pd.read_csv("dataset_savioroftheheart.csv")
    st.dataframe(data.head())

    Age= st.number_input(label = "Enter your age",min_value=0,max_value=100,step=1)
    Sex = st.selectbox('Sex',('Male','Female'))
    Smoking = st.selectbox('Do you smoke?',('Yes','No'))
    Diabetes=st.selectbox('Do you have diabetes?',('Yes','No'))
    Hypertension_treatment = st.selectbox('Are you undergoing any treatment for hypertension?',('Yes','No'))
    HDL = st.number_input(label ='Enter your HDL level',min_value=10,max_value = 100,step=1)
    TCL = st.number_input(label ="Enter your TCL level",min_value=100,max_value=405,step=1)
    SBP = st.number_input(label = "Enter your SBP value",min_value=20,max_value=200,step=1)
    result = ""

    if st.button('Predict'):
        result = prediction(Age,Sex,Smoking,Diabetes,Hypertension_treatment,HDL,TCL,SBP)
        st.success(result)


if __name__=="__main__":
    render_prediction()






