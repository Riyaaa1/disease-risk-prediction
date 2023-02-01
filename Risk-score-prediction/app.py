import pickle
import streamlit as st
import pandas as pd

# loading the trained model 
pickle_in = open('regressor.pkl','rb')
regressor = pickle.load(pickle_in)

def prediction(Age,Sex,Smoking,Diabetes,Hypertension_treatment,HDL,TCL,SBP):
    if Sex == 'Male':
        Sex = 1
    else:
        Sex = 0


    if Smoking == 'Yes':
        Smoking = 1
    else:
        Smoking = 0

    if Diabetes == 'Yes':
        Diabetes = 1
    else:
        Diabetes = 0

    if Hypertension_treatment == 'Yes':
        Hypertension_treatment = 1
    else:
        Hypertension_treatment = 0

    #making predictions
    prediction = regressor.predict(
        [[Age,Sex,Smoking,Diabetes,Hypertension_treatment,HDL,TCL,SBP]])
    print(prediction)
    if prediction >= 0.5:
        pred = 'High risk'
    else:
        pred = 'Low risk'
    return prediction

# this is the main function in which we define our webpage  
def main():       
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
    main()






