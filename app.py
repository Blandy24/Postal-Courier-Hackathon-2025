import streamlit as st
import pandas as pd
import numpy as np
from haversine import haversine, Unit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import matplotlib.pyplot as plt

# -----------------------------
# 1. Streamlit Config
# -----------------------------
st.set_page_config(page_title="PostaConnect", layout="wide")

st.title("📦 PostaConnect - AI-powered Delivery Intelligence System")
st.markdown("Participant : **Blandina Kakore**")

# -----------------------------
# 2. Upload or Load Dataset
# -----------------------------
uploaded_file = st.sidebar.file_uploader("Upload delivery dataset", type=["csv"])

if uploaded_file:
    posta = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset: DATA/delivery_five_cities_tanzania.csv")
    posta = pd.read_csv(r"DATA\delivery_five_cities_tanzania.csv")

# -----------------------------
# 3. Preprocessing
# -----------------------------
posta['sign_time'] = pd.to_datetime("2025-" + posta['sign_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
posta['receipt_time'] = pd.to_datetime("2025-" + posta['receipt_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

posta['delivery_duration'] = (posta['sign_time'] - posta['receipt_time']).dt.total_seconds() / 60
posta['time_of_day'] = posta['receipt_time'].dt.hour
posta['day_of_week'] = posta['receipt_time'].dt.dayofweek

posta['distance'] = posta.apply(
    lambda row: haversine(
        (row['poi_lat'], row['poi_lng']),
        (row['receipt_lat'], row['receipt_lng']),
        unit=Unit.METERS
    ),
    axis=1
)

# -----------------------------
# 4. Train Model
# -----------------------------
X = posta[['time_of_day', 'day_of_week', 'distance', 'poi_lat', 'poi_lng', 'receipt_lat', 'receipt_lng']]
y = posta['delivery_duration']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -----------------------------
# 5. Anomaly Detection
# -----------------------------
iso_forest = IsolationForest(contamination=0.05, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_test)

X_test = X_test.copy()
X_test['delivery_duration'] = y_test
X_test['prediction'] = y_pred
X_test['anomaly'] = anomaly_labels

# -----------------------------
# 6. Dashboard Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Overview",
    "⏱ Duration Distribution",
    "🤖 Model Predictions",
    "🚨 Anomaly Detection"
])

with tab1:
    st.subheader("Dataset Overview")
    st.write(posta.head())
    st.metric("Number of records", len(posta))
    st.metric("Number of anomalies detected", (X_test['anomaly'] == -1).sum())
    st.metric("Model MAE (minutes)", f"{mae:.2f}")
    st.metric("Model R²", f"{r2:.2f}")

with tab2:
    st.subheader("Delivery Duration Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(posta['delivery_duration'].dropna(), bins=50, color="skyblue", edgecolor="black")
    ax.set_xlabel("Delivery Duration (minutes)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with tab3:
    st.subheader("Model Predictions vs Actual")
    fig_pred = px.scatter(
        X_test,
        x="delivery_duration",
        y="prediction",
        title="Actual vs Predicted Delivery Time",
        trendline="ols"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

with tab4:
    st.subheader("Anomaly Detection")
    
    # Scatter plot: anomalies vs predictions
    fig_anom = px.scatter(
        X_test,
        x="delivery_duration",
        y="prediction",
        color=X_test['anomaly'].map({-1: "Anomaly", 1: "Normal"}),
        title="Anomalies Highlighted",
        labels={"color": "Anomaly Status"}
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # --- Map Visualization ---
    st.subheader("🗺 Anomalies per City (Tanzania Map)")

    # Tanzania city coordinates (approx centers)
    tanzania_cities = {
        "Dar es Salaam": (-6.7924, 39.2083),
        "Dodoma": (-6.1630, 35.7516),
        "Arusha": (-3.3869, 36.6829),
        "Mwanza": (-2.5164, 32.9175),
        "Mbeya": (-8.9090, 33.4608),
    }

    # Assign nearest city based on receipt coordinates
    def assign_city(lat, lng):
        min_city, min_dist = None, float("inf")
        for city, coords in tanzania_cities.items():
            dist = haversine((lat, lng), coords, unit=Unit.KILOMETERS)
            if dist < min_dist:
                min_city, min_dist = city, dist
        return min_city

    map_data = X_test.copy()
    map_data["lat"] = posta.loc[X_test.index, "receipt_lat"]
    map_data["lng"] = posta.loc[X_test.index, "receipt_lng"]
    map_data["city"] = map_data.apply(lambda row: assign_city(row["lat"], row["lng"]), axis=1)
    map_data["order_id"] = posta.loc[X_test.index, "order_id"]

    # --- Table of anomalies per city ---
    anomalies_only = map_data[map_data['anomaly'] == -1]
    city_count_table = anomalies_only.groupby("city").agg(
        anomaly_count=('order_id','count'),
        order_ids=('order_id', lambda x: ", ".join(map(str,x)))
    ).reset_index()
    
    # --- Add download button instead ---
    st.subheader("📥 Download Anomalies Data")
    csv_anomalies = anomalies_only.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download anomalies CSV",
        data=csv_anomalies,
        file_name='anomalies_data.csv',
        mime='text/csv'
        )

    # --- Expandable list per city ---
    for city, group in anomalies_only.groupby("city"):
        with st.expander(f"{city} - {len(group)} anomalies"):
            st.write("Order IDs:")
            st.write(group['order_id'].tolist())

    # --- Map Plot ---
    fig_map = px.scatter_map(
        map_data,
        lat="lat",
        lon="lng",
        color=map_data['anomaly'].map({-1: "Anomaly", 1: "Normal"}),
        hover_name="city",
        hover_data={"delivery_duration": True, "prediction": True, "order_id": True},
        height=700,
        title="Anomalies by Location"
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": -6.0, "lon": 35.5},  # center of Tanzania
        mapbox_zoom=5.5,  # fit all regions
        legend_title="Delivery Status",
        margin={"r":0,"t":30,"l":0,"b":0}
    )

    st.plotly_chart(fig_map, use_container_width=True)
