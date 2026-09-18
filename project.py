import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# -----------------------------
# SETTINGS
# -----------------------------
RANDOM_STATE = 42

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# -----------------------------
# LOAD / GENERATE DATA
# -----------------------------
@st.cache_data
def load_data():

    np.random.seed(RANDOM_STATE)

    n_houses = 400

    locations = [
        "Downtown",
        "Suburb",
        "Rural",
        "Beachside"
    ]

    location_price_factor = {
        "Downtown": 1.6,
        "Beachside": 1.5,
        "Suburb": 1.1,
        "Rural": 0.75
    }

    # Generate house details
    area_sqft = np.round(
        np.random.normal(1800, 600, n_houses).clip(400, 5000),
        0
    )

    bedrooms = np.random.randint(1, 6, n_houses)

    bathrooms = np.random.randint(1, 4, n_houses)

    age_years = np.random.randint(0, 40, n_houses)

    location = np.random.choice(
        locations,
        n_houses
    )

    # Calculate base price
    base_price = (
        area_sqft * 120
        + bedrooms * 8000
        + bathrooms * 5000
        - age_years * 1000
    )

    # Apply location factor
    price = base_price * np.array([
        location_price_factor[loc]
        for loc in location
    ])

    # Add random noise
    price = price + np.random.normal(
        0,
        25000,
        n_houses
    )

    # Minimum price = 30,000
    price = np.round(
        price.clip(30000, None),
        -2
    )

    # Create dataframe
    df = pd.DataFrame({
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "location": location,
        "price": price
    })

    return df


# -----------------------------
# TRAIN MODELS
# -----------------------------
@st.cache_resource
def train_models(df):

    df = df.copy()

    # Feature engineering
    df["total_rooms"] = (
        df["bedrooms"] + df["bathrooms"]
    )

    df["price_per_sqft_est"] = (
        df["area_sqft"]
        / df["total_rooms"].replace(0, 1)
    )

    # Convert location into numerical columns
    df_encoded = pd.get_dummies(
        df,
        columns=["location"],
        drop_first=True
    )

    # Input features
    X = df_encoded.drop(
        columns=["price"]
    )

    # Target
    y = df_encoded["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    # -------------------------
    # STANDARD SCALER
    # -------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -------------------------
    # LINEAR REGRESSION
    # -------------------------
    lin_model = LinearRegression()

    lin_model.fit(
        X_train_scaled,
        y_train
    )

    lin_pred = lin_model.predict(
        X_test_scaled
    )

    # -------------------------
    # RANDOM FOREST
    # -------------------------
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_pred = rf_model.predict(
        X_test
    )

    # -------------------------
    # MODEL METRICS
    # -------------------------
    metrics = {

        "Linear Regression": {
            "MAE": mean_absolute_error(
                y_test,
                lin_pred
            ),
            "R2": r2_score(
                y_test,
                lin_pred
            )
        },

        "Random Forest": {
            "MAE": mean_absolute_error(
                y_test,
                rf_pred
            ),
            "R2": r2_score(
                y_test,
                rf_pred
            )
        }
    }

    return (
        lin_model,
        rf_model,
        scaler,
        X.columns,
        y_test,
        lin_pred,
        rf_pred,
        metrics
    )


# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("🏠 House Price Prediction System")

st.write(
    "Enter house details on the left to get a predicted price."
)


# -----------------------------
# LOAD DATA AND TRAIN MODELS
# -----------------------------
df = load_data()

(
    lin_model,
    rf_model,
    scaler,
    feature_columns,
    y_test,
    lin_pred,
    rf_pred,
    metrics
) = train_models(df)


# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🏠 House Details")

area_sqft = st.sidebar.slider(
    "Area (sqft)",
    400,
    5000,
    2000,
    step=50
)

bedrooms = st.sidebar.slider(
    "Bedrooms",
    1,
    5,
    3
)

bathrooms = st.sidebar.slider(
    "Bathrooms",
    1,
    3,
    2
)

age_years = st.sidebar.slider(
    "Age of house (years)",
    0,
    40,
    5
)

location = st.sidebar.selectbox(
    "Location",
    [
        "Downtown",
        "Suburb",
        "Rural",
        "Beachside"
    ]
)

model_choice = st.sidebar.radio(
    "Select Model",
    [
        "Random Forest",
        "Linear Regression"
    ]
)


# -----------------------------
# CREATE NEW HOUSE DATA
# -----------------------------
new_house = pd.DataFrame([
    {
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
        "location": location
    }
])


# Feature engineering
new_house["total_rooms"] = (
    new_house["bedrooms"]
    + new_house["bathrooms"]
)

new_house["price_per_sqft_est"] = (
    new_house["area_sqft"]
    / new_house["total_rooms"]
)


# Encode location
new_house_encoded = pd.get_dummies(
    new_house,
    columns=["location"],
    drop_first=True
)


# Make sure columns exactly match training data
new_house_encoded = new_house_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)


# -----------------------------
# PREDICTION
# -----------------------------
if model_choice == "Random Forest":

    predicted_price = rf_model.predict(
        new_house_encoded
    )[0]

else:

    new_house_scaled = scaler.transform(
        new_house_encoded
    )

    predicted_price = lin_model.predict(
        new_house_scaled
    )[0]


# -----------------------------
# DISPLAY PREDICTION
# -----------------------------
st.subheader("💰 Predicted Price")

st.markdown(
    f"## ${predicted_price:,.0f}"
)


# -----------------------------
# MODEL PERFORMANCE
# -----------------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Random Forest R²",
        f"{metrics['Random Forest']['R2']:.3f}"
    )

    st.metric(
        "Random Forest MAE",
        f"${metrics['Random Forest']['MAE']:,.0f}"
    )


with col2:

    st.metric(
        "Linear Regression R²",
        f"{metrics['Linear Regression']['R2']:.3f}"
    )

    st.metric(
        "Linear Regression MAE",
        f"${metrics['Linear Regression']['MAE']:,.0f}"
    )


# -----------------------------
# ACTUAL VS PREDICTED GRAPH
# -----------------------------
st.subheader(
    "📈 Actual vs Predicted Prices (Test Set)"
)

fig, ax = plt.subplots(
    figsize=(6, 5)
)

ax.scatter(
    y_test,
    rf_pred,
    alpha=0.6,
    label="Random Forest"
)

ax.scatter(
    y_test,
    lin_pred,
    alpha=0.4,
    label="Linear Regression"
)

# Find graph limits
lims = [
    min(
        y_test.min(),
        rf_pred.min(),
        lin_pred.min()
    ),
    max(
        y_test.max(),
        rf_pred.max(),
        lin_pred.max()
    )
]

# Perfect prediction line
ax.plot(
    lims,
    lims,
    "k--",
    label="Perfect Prediction"
)

ax.set_xlabel(
    "Actual Price"
)

ax.set_ylabel(
    "Predicted Price"
)

ax.set_title(
    "Actual vs Predicted House Prices"
)

ax.legend()

st.pyplot(fig)


# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)


# -----------------------------
# AVERAGE PRICE BY LOCATION
# -----------------------------
st.subheader(
    "🏙️ Average Price by Location"
)

avg_price = (
    df.groupby("location")["price"]
    .mean()
    .sort_values()
)

st.bar_chart(
    avg_price
)


# -----------------------------
# SELECTED HOUSE DETAILS
# -----------------------------
st.subheader("🏡 Selected House Details")

details = pd.DataFrame({
    "Feature": [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age",
        "Location",
        "Selected Model"
    ],

    "Value": [
        f"{area_sqft} sqft",
        bedrooms,
        bathrooms,
        f"{age_years} years",
        location,
        model_choice
    ]
})

st.table(details)
