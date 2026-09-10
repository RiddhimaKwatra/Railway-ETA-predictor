#Importing required modules
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
st.set_page_config(page_title="Railway ETA predictor", page_icon="🚇")
st.title("Railway ETA predictor")

mymodel = joblib.load('model.pkl')
One_Hot_Encoder = joblib.load('encoder.pkl')

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #131824 50%, #0d1520 100%);
        color: #e0e0e0;
}
    }
    [data-testid="stMetric"] {
        background-color: #131824;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b96a5;
    }
    [data-testid="stMetricValue"] {
        color: #22d3ee;
    }
    .stSelectbox label, .stNumberInput label {
        color: #8b96a5 !important;
    }
    h1 {
        color: #ffffff;
    }
    .stButton button {
        background-color: #22d3ee;
        color: #0a0e1a;
        border: none;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


#['"", 'Station_Code', 'Train_Priority', 'Weather_Condition',
       #'Time_of_Day', 'Day_of_Week', 'Network_Congestion', 'Active_Incident',
       #'Scheduled_Arrival_Mins', 'Actual_Arrival_Mins', 'Delay_Minutes']

def get_val():
    st.subheader('Enter  the following:')
    try:
        Station_Code=str(st.text_input("Station_Code:", "NDLS"))
        Train_Priority=str(st.text_input("Train_Priority:", "FREIGHT"))
        Weather_Condition=str(st.text_input("Weather_Condition","CLEAR"))
        Time_of_Day=str(st.text_input("Time_of_Day","EVENING"))
        Day_of_Week=str(st.text_input("Day_of_Week","SATURDAY"))
        Network_Congestion=str(st.text_input("Network_Congestion","HIGH"))
        Active_Incident=str(st.text_input("Active_Incident","NO"))
        Scheduled_Arrival_Mins=int(st.text_input("Scheduled_Arrival_Mins:", 807))
    except ValueError:
        st.error("⚠️ Invalid input!")
        return None

    input_dict = {
        'Station_Code': Station_Code,
        'Train_Priority': Train_Priority,
        'Weather_Condition': Weather_Condition,
        'Time_of_Day': Time_of_Day,
        'Day_of_Week': Day_of_Week,
        'Network_Congestion': Network_Congestion,
        'Active_Incident': Active_Incident,
        'Scheduled_Arrival_Mins': Scheduled_Arrival_Mins
    }
    st.subheader('Your entered values')
    st.dataframe(pd.DataFrame([input_dict]))
    st.markdown('Press this button to predict your results')
    Predbutton=st.button('PREDICT')
    if Predbutton:
        with st.spinner("Imputing values... One Hot Encoding... Predicting...", show_time=True):
            time.sleep(3)
        st.success(predict_eta(pd.DataFrame([input_dict])))
        st.markdown(f'Railway ETA Predictor is subject to errors with an MAE of ~8 mins')

def process_input(data):
    cato_cols = list(One_Hot_Encoder.feature_names_in_)
    #numerical cols
    num_cols = ['Scheduled_Arrival_Mins']
    X = pd.DataFrame(data[num_cols], columns=num_cols)
    #categorical cols
    OH_cols_X = pd.DataFrame(One_Hot_Encoder.transform(data[cato_cols]))
    OH_cols_X.index = data.index
    #join cols
    joined=pd.concat([X, OH_cols_X], axis=1)
    joined.columns=joined.columns.astype(str)
    return joined, data['Scheduled_Arrival_Mins'].iloc[0]
def predict_eta(_):
    X_input, Scheduled_Arrival_Mins  = process_input(_)
    pred = mymodel.predict(X_input)
    return f"\n Predicted ETA: {Scheduled_Arrival_Mins+int(pred[0])}"

get_val()       

