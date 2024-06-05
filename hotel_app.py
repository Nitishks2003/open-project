import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load your data
data = pd.read_csv("C:/Users/User.ACIES35/Desktop/code/Sample_cleaned_features.csv")  # Replace "your_data.csv" with the path to your data file

# Split features and target variable
X = data.drop("is_canceled", axis=1)  # Replace "your_target_column" with the name of your target column
y = data["is_canceled"]

# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Define the Streamlit app
def main():
    st.title("Hotel Booking Prediction App")
    st.sidebar.header("User Input Features")

    # Collect user input features
    lead_time = st.sidebar.slider("Lead Time", min_value=0, max_value=365, value=50)
    arrival_month = st.sidebar.selectbox("Arrival Month", range(1, 13), index=0)
    adults = st.sidebar.number_input("Number of Adults", min_value=1, max_value=10, value=1)
    children = st.sidebar.number_input("Number of Children", min_value=0, max_value=10, value=0)
    stays_weekend_nights = st.sidebar.slider("Stays in Weekend Nights", min_value=0, max_value=10, value=0)
    stays_week_nights = st.sidebar.slider("Stays in Week Nights", min_value=0, max_value=20, value=0)
    # Add more input fields for other features
    
    # Create a dictionary to hold user input
    user_input = {
        'lead_time': lead_time,
        'arrival_date_month': arrival_month,
        'adults': adults,
        'children': children,
        'stays_in_weekend_nights': stays_weekend_nights,
        'stays_in_week_nights': stays_week_nights,
    }

    # Convert user input into DataFrame
    input_df = pd.DataFrame([user_input])

    # Ensure input_df has the same columns as X
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = rf_model.predict(input_df)
    prediction_proba = rf_model.predict_proba(input_df)

    # Display prediction
    st.subheader("Prediction")
    if prediction[0] == 1:
        st.write("Booking will be canceled.")
    else:
        st.write("Booking will not be canceled.")

    # Display prediction probabilities
    st.subheader("Prediction Probability")
    st.write(f"Probability of not canceling: {prediction_proba[0][0]:.2f}")
    st.write(f"Probability of canceling: {prediction_proba[0][1]:.2f}")

    # Plot prediction probabilities
    st.subheader("Prediction Probability Visualization")
    prob_df = pd.DataFrame(prediction_proba, columns=["Not Canceled", "Canceled"])
    st.bar_chart(prob_df.T)

    # Display input features
    st.subheader("Input Features")
    st.write(input_df)

if __name__ == "__main__":
    main()