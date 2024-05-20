from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  # Add this line
import math 
import streamlit as st
import joblib
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
import joblib

def get_data():
# Read CSV file
    df = pd.read_csv('C:/Users/alexi/OneDrive/VAWI/PA2/Classification/data.csv')

# Data (as pandas dataframes)
    X = df.drop('Class', axis=1)  # replace 'target_column' with the name of your target column
    y = df['Class']  # replace 'target_column' with the name of your target column

    print(df.head())
    return df

def train_and_predict(df):
    # Split the data into features and target variable
    X = df.drop('Class', axis=1)
    y = df['Class']

    #scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of LogisticRegression
    model = LogisticRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # testmodel
    y_pred = model.predict(X_test)

    # Print the classification report
    print("classification_report:", classification_report(y_test, y_pred))
    # Print the accuracy score
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return model, scaler


def main():
        data= get_data()
        model, scaler = train_and_predict(data)  
        # Save the model to file
        joblib.dump(model, 'trainmodel/model.pkl')
        joblib.dump(scaler, 'trainmodel/scaler.pkl')
        print(type(model))

if __name__ == '__main__':
    main()
