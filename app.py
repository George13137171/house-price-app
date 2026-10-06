from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pandas as pd 
import streamlit as st

try:

    housing = fetch_california_housing(as_frame = True)
    df = housing.frame
    print(df.head())

    y = df['MedHouseVal']
    X = df.drop(columns = ['MedHouseVal'])

    model = LinearRegression()
    model.fit(X, y)

    sample_house = X.iloc[[0]]
    price = model.predict(sample_house)
    print(f"Predicted price for the sample house: {price[0]}")





except Exception as e:
    print(f"An error occured: {e}")

user_inputs = {}
for col in X.columns:
    user_inputs[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))

if st.button("Predict Price"):
    try:
        user_input_df = pd.DataFrame([user_inputs])
        predicted_price = model.predict(user_input_df)
        st.write(f"Predicted price for the house: {predicted_price[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

