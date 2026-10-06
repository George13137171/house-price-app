
from sklearn.linear_model import LinearRegression
import pandas as pd 
import streamlit as st

link = st.text_input("Enter the link to the CSV file:", "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")
if link:

    try:


        df = pd.read_csv(link)
        df = df.dropna()
        target = st.selectbox("Select the target variable:", df.columns)

        y = df[target]
        X = df.drop(columns = [target])
        X_num = X.select_dtypes(include=['number'])
        model = LinearRegression()
        model.fit(X_num, y)






    except Exception as e:
        print(f"An error occured: {e}")

user_inputs = {}
for col in X_num.columns:
    user_inputs[col] = st.number_input(label= f"Enter value for {col}:", value = float (X_num[col].mean()))

if st.button("Predict Price"):
    try:
        user_input_df = pd.DataFrame([user_inputs])
        predicted_price = model.predict(user_input_df)
        st.write(f"Predicted price for the house: {predicted_price[0]}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

