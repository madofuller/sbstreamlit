import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Streamlit UI
st.title('Baseball Betting Algorithm')

# Upload CSV file
st.subheader('Upload CSV File')
uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])

# Load and process data
if uploaded_file is not None:
    st.write('Uploaded file:')
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # Data preprocessing
    X = df.drop(['TargetColumn'], axis=1)  # Adjust 'TargetColumn' to the actual target column name
    y = df['TargetColumn']  # Adjust 'TargetColumn' to the actual target column name

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    st.subheader('Train the Model')
    n_estimators = st.number_input('Number of Estimators', min_value=1, value=100)
    learning_rate = st.number_input('Learning Rate', min_value=0.001, max_value=1.0, value=0.1)

    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)

    st.write('Model trained!')

    # Model evaluation
    st.subheader('Evaluate the Model')
    test_score = model.score(X_test, y_test)
    st.write(f'Model R^2 Score on Test Data: {test_score:.2f}')

    # Prediction
    st.subheader('Make Predictions')
    example_input = {}  # Create an example input dictionary based on your dataset
    for column in X.columns:
        example_input[column] = st.number_input(f'Enter {column}', value=0.0)

    prediction = model.predict(pd.DataFrame([example_input]))
    st.write(f'Predicted Value: {prediction[0]:.2f}')
