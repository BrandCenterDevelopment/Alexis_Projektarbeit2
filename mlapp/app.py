import streamlit as st 
import pandas as pd
import joblib
import numpy as np  
from ucimlrepo import fetch_ucirepo
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.externals import joblib    # Add this line


def get_data():
    # fetch dataset 
    df = fetch_ucirepo(id=545) 
    # data (as pandas dataframes) 
    X = df.data.features 
    y = df.data.targets
    df = pd.concat([X, y], axis=1)
    return df

def sidebar():
    st.sidebar.header("Class Predictor sidebar")
    data = get_data()
    features = {}
    slider_labels = data.columns[:-1]  # Assumes the last column is the target

    for label in slider_labels:
        min_val = float(data[label].min())  # Convert to float
        max_val = float(data[label].max())  # Convert to float
        default_val = float(data[label].mean())  # Convert to float
        step = (max_val - min_val) / 100  # Calculate step size as float

        # Ensure step is not zero to avoid another potential error when range is zero
        if step == 0:
            step = 0.01  # Assign a small float value if range is zero

        features[label] = st.sidebar.slider(label, min_val, max_val, default_val, step)
    return features

def use_scaledvalues(input_dict):
    data = get_data()
    # Scale the input data
    scaled_values = {}
    for key, value in input_dict.items():
        min_val = data[key].min()
        max_val = data[key].max()
        scaled_values[key] = (value - min_val) / (max_val - min_val)
    return scaled_values

def get_radar_chart(input_data):
    st.write("test")
    # Create a list of feature names
    feature_names = list(input_data.keys())
    print("feature_names:", feature_names)
    # Create a list of feature values
    feature_values = list(use_scaledvalues(input_data).values())
    # Create a radar chart
    fig = go.Figure(data=go.Scatterpolar(r=feature_values, theta=feature_names, fill='toself'))
    # Update the layout of the radar chart
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False,width=800, height=700)
    # Display the radar chart
    #st.plotly_chart(fig)
    return fig

def userprediction(input_data):
    # Load the model and scaler
    model = joblib.load('trainmodel/model.pkl')
    scaler = joblib.load('trainmodel/scaler.pkl')

    # Convert input dictionary to DataFrame to maintain feature names
    feature_df = pd.DataFrame([input_data])  # Convert dict to DataFrame, keys become column names
        #feature_df = np.array(list(input_data.values())).reshape(1, -1) # Convert dict to DataFrame, keys become column names
    st.write("feature_df:", feature_df)    

    # Scale the feature values while preserving column names
    feature_values_scaled = scaler.transform(feature_df)  # DataFrame maintains column names

    # Make a prediction
    prediction = model.predict(feature_values_scaled)

    st.write("feature_values_scaled:", feature_values_scaled  )
    st.write(prediction)
    st.write("The predicted class is:", prediction[0])

    return prediction
    
def main():
    st.set_page_config(
        page_title="Class Predictor", 
        page_icon=   "",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    input_data=sidebar()
    with st.container():
        st.title('Class Predictor')  # Add a title  # Add a title
        st.write("test")
    col1, col2 = st.columns([4,1])
   
    with col1:
        radarchart = get_radar_chart(input_data)
        st.plotly_chart(radarchart)

    with col2:
        userprediction(input_data)
  

if __name__ == '__main__':
    main()  

