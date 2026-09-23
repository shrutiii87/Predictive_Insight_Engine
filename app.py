import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import (
    model,
    data,
    features,
    mae,
    rmse,
    r2
)


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Predictive Insight Engine",
    page_icon="🏠",
    layout="wide"
)


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🏠 Predictive Insight Engine")

st.subheader(
    "House Price Prediction using Multiple Linear Regression"
)

st.write(
    "Enter the property details below to estimate "
    "the predicted market price."
)


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.header("🏠 Property Details")


area = st.sidebar.number_input(
    "Area (sqft)",
    min_value=300,
    max_value=10000,
    value=1500,
    step=100
)


bedrooms = st.sidebar.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)


bathrooms = st.sidebar.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)


location_score = st.sidebar.slider(
    "Location Score",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1
)


age_years = st.sidebar.number_input(
    "House Age (years)",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)


# -------------------------------------------------
# PREDICTION
# -------------------------------------------------

if st.sidebar.button(
    "🔮 Predict House Price",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            area,
            bedrooms,
            bathrooms,
            location_score,
            age_years
        ]],
        columns=features
    )


    prediction = model.predict(
        input_data
    )[0]


    st.success(
        "Prediction generated successfully!"
    )


    st.metric(
        "Predicted House Price",
        f"₹ {prediction:,.0f}"
    )


# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

st.header("📊 Model Performance")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "R² Score",
        f"{r2:.4f}"
    )


with col2:

    st.metric(
        "MAE",
        f"₹ {mae:,.0f}"
    )


with col3:

    st.metric(
        "RMSE",
        f"₹ {rmse:,.0f}"
    )


# -------------------------------------------------
# MODEL INFORMATION
# -------------------------------------------------

st.header("🧠 Model Information")


col1, col2 = st.columns(2)


with col1:

    st.write("### Input Features")

    st.write(
        """
        - Area (sqft)
        - Bedrooms
        - Bathrooms
        - Location Score
        - Age of House
        """
    )


with col2:

    st.write("### Algorithm")

    st.write(
        """
        **Multiple Linear Regression**

        The model learns the relationship between
        property features and house price.
        """
    )


# -------------------------------------------------
# DATASET
# -------------------------------------------------

st.header("📂 Dataset")


st.write(
    f"Total Records: **{len(data):,}**"
)


st.dataframe(
    data.head(20),
    use_container_width=True
)


# -------------------------------------------------
# ACTUAL VS PREDICTED
# -------------------------------------------------

st.header("📈 Actual vs Predicted Price")


from sklearn.model_selection import train_test_split


X = data[features]
y = data["house_price_inr"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


predictions = model.predict(X_test)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.scatter(
    y_test,
    predictions,
    alpha=0.5
)


min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)


ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)


ax.set_xlabel(
    "Actual House Price"
)

ax.set_ylabel(
    "Predicted House Price"
)

ax.set_title(
    "Actual vs Predicted House Prices"
)


st.pyplot(fig)


# -------------------------------------------------
# FEATURE COEFFICIENTS
# -------------------------------------------------

st.header("📌 Model Coefficients")


coefficients = pd.DataFrame({

    "Feature": features,

    "Coefficient": model.coef_

})


st.dataframe(
    coefficients,
    use_container_width=True,
    hide_index=True
)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "Predictive Insight Engine | Supervised Learning | "
    "Multiple Linear Regression"
)
