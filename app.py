from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import pandas as pd 

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
