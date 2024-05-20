import streamlit as st 
import pandas as pd
import joblib
import numpy as np  
#from ucimlrepo import fetch_ucirepo
from scipy.io.arff import loadarff
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#from sklearn.externals import joblib    # Add this line

@st.cache_data 
def get_data():
    # Read CSV file
    df = pd.read_csv('data.csv')

    # Drop the 'perimeter' column
    df = df.drop('Perimeter', axis=1)

    # Data (as pandas dataframes)
    X = df.drop('Class', axis=1)  # replace 'target_column' with the name of your target column
    y = df['Class']  # replace 'target_column' with the name of your target column

    print(df.head())
    return df

def sidebar():
    st.sidebar.header("Slide the values")
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
    
    # Create a list of feature names
    feature_names = list(input_data.keys())
    print("feature_names:", feature_names)
    # Create a list of feature values
    feature_values = list(use_scaledvalues(input_data).values())
    # Create a radar chart
    fig = go.Figure(data=go.Scatterpolar(r=feature_values, theta=feature_names, fill='toself'))
    # Update the layout of the radar chart
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        width=1200, 
        height=700,
        font=dict(
            family="Courier New, monospace",
            size=15,
            color="black"
        )
    )
    # Display the radar chart
    #st.plotly_chart(fig)
    return fig


def add_predictions(input_data):
    # Load the model and scaler
    model = joblib.load('trainmodel/model.pkl')
    scaler = joblib.load('trainmodel/scaler.pkl')
    # Convert input_data values to a 2D array
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    #st.write("input_array:", input_array)
    # Scale the input data with scaler
    scaled_input = scaler.transform(input_array)
    #st.write("scaled_input:", scaled_input) 
    # Make predictions
  
    prediction = model.predict(scaled_input)
    # Display the prediction
    
    st.write("")
    st.markdown(f'<h2 style="font-weight: bold;">Prediction: {prediction[0]}</h2>', unsafe_allow_html=True)
    if prediction[0] == "Cammeo":
        st.markdown(f'<p style="color:green;">Probability for Cammeo: {model.predict_proba(scaled_input)[0][0]:.2f}**</p>', unsafe_allow_html=True)
    else:
        st.markdown(f"Probability for Cammeo: {model.predict_proba(scaled_input)[0][0]:.2f}")
    if prediction[0] == "Osmancik":
        st.markdown(f'<p style="color:green;">Probability for Osmancik: {model.predict_proba(scaled_input)[0][1]:.2f}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f"**Probability for Osmancik: {model.predict_proba(scaled_input)[0][1]:.2f}**")
    st.write("This app allows you to predict the class of rice grain based on 7 morphological features. \n\n Please read the project paper for further explanation. \n\n A total of 3810 rice grain's images were taken for the two species, processed and feature inferences were made. \n\n 7 morphological features were obtained for each grain of rice.")       
  

def main():
    st.set_page_config(
        page_title="Class Predictor", 
        page_icon='🌾',
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("mlapp/css/style.css") as f: #f for file
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data=sidebar()
    with st.container():
        st.title('Class Predictor')  # Add a title  # Add a title
        st.write("This application harnesses the power of binary logistic regression to classify rice plants into distinct categories based on seven key features. Through a user-friendly interface, users can input data related to morphologic features . This model processes this information to predict the class with high accuracy.\n\n \n\n \n\n ")
        st.write("")
        st.write("")
        st.write("")
    col1, col2 = st.columns([4,1])
   
    with col1:
        radarchart = get_radar_chart(input_data)
        st.plotly_chart(radarchart)

    with col2:
      
        add_predictions(input_data)
  

if __name__ == '__main__':
    main()  

