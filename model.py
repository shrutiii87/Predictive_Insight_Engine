import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


# Load dataset
data = pd.read_csv(
    "RealEstate_HousePrice_Dataset_4200.csv"
)


# Features
features = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "location_score",
    "age_years"
]

target = "house_price_inr"


# X and y
X = data[features]
y = data[target]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train model
model = LinearRegression()

model.fit(
    X_train,
    y_train
)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)
