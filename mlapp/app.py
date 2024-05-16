import streamlit as st 
import pandas as pd
import joblib
import numpy as np  
from ucimlrepo import fetch_ucirepo
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import io


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
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False, width=800, height=700)
    # Display the radar chart
    #st.plotly_chart(fig)
    return fig

def userprediction(input_data):
    # Load the model and scaler
    try:
        model = joblib.load('trainmodel/model.pkl')
        scaler = joblib.load('trainmodel/scaler.pkl')
    except FileNotFoundError as e:
        st.write("Error loading model or scaler:", e)
        return

    # Convert input dictionary to DataFrame to maintain feature names
    feature_df = pd.DataFrame([input_data])  # Convert dict to DataFrame, keys become column names
    st.write("feature_df:", feature_df)    

    # Scale the feature values while preserving column names
    try:
        feature_values_scaled = scaler.transform(feature_df)  # DataFrame maintains column names
    except Exception as e:
        st.write("Error scaling feature values:", e)
        return

    # Make a prediction
    try:
        prediction = model.predict(feature_values_scaled)
    except Exception as e:
        st.write("Error making prediction:", e)
        return

    st.write("feature_values_scaled:", feature_values_scaled)
    st.write("Prediction:", prediction)
    st.write("The predicted class is:", prediction[0])

    return prediction

import pandas as pd
import numpy as np
import joblib

import pandas as pd
import numpy as np
import joblib
import io  # Make sure to import the io module

def test_predictions():
    # Correctly formatted sample test data in CSV format with matching column names as during training
    test_data = """
    Area,Perimeter,Major_Axis_Length,Minor_Axis_Length,Eccentricity,Convex_Area,Extent, class
    12068,457.7390137,196.4420624,79.54393768,0.914350748,12347,0.599503219,Cammeo
    15710,517.0700073,214.7502136,95.24736023,0.896261394,16259,0.607595921,Cammeo
    15545,510.2839966,213.4366455,93.55502319,0.898815632,15909,0.568165183,Cammeo
    12720,449.6239929,178.6147766,92.7516861,0.854602098,13095,0.743859649,Cammeo
    15073,498.3619995,205.8031006,93.99627686,0.889605761,15476,0.613921463,Cammeo
    11771,441.8599854,186.9977722,81.26961517,0.900622606,12109,0.619265556,Cammeo
    14873,499.6159973,210.2774963,91.58361816,0.900170863,15195,0.573030233,Cammeo
    """
    # Parse the string into a DataFrame correctly using pd.read_csv
    feature_df = pd.read_csv(io.StringIO(test_data))

    # Load the model and scaler
    try:
        model = joblib.load('trainmodel/model.pkl')
        scaler = joblib.load('trainmodel/scaler.pkl')
    except Exception as e:
        st.write(f"Error loading model or scaler: {e}")
        return  # Exit if the model or scaler cannot be loaded

    st.write("feature_df:", feature_df)

    # Scale the feature values while preserving column names
    try:
        feature_values_scaled = scaler.transform(feature_df)
    except Exception as e:
        st.write("Error scaling feature values:", e)
        return

    # Make a prediction
    try:
        prediction = model.predict(feature_values_scaled)
    except Exception as e:
        st.write("Error making prediction:", e)
        return

    st.write("feature_values_scaled:", feature_values_scaled)
    st.write("Prediction:", prediction)
    st.write("The predicted class is:", prediction[0])

    return prediction




def main():
    st.set_page_config(
        page_title="Class Predictor", 
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    input_data = sidebar()
    with st.container():
        st.title('Class Predictor')  # Add a title
        st.write("test")
    col1, col2 = st.columns([4, 1])

    with col1:
        radarchart = get_radar_chart(input_data)
        st.plotly_chart(radarchart)

    with col2:
        userprediction(input_data)
        test_predictions()

if __name__ == '__main__':
    main()

