# Team Name
Blandina Kakore

# Team Members
Blandina Kakore

# Problem
Develop an AI-powered Delivery Intelligence System that predicts parcel delivery completion time and detects anomalies in delivery patterns, such as unusually long or short delivery times that may indicate fraud or operational issues.

# Solution Overview
This project develops a Delivery Intelligence System using the provided courier dataset.
The system includes:
- **Linear Regression** to predict delivery completion time based on features like time of day, day of week, distance, and coordinates.
- **Isolation Forest** for anomaly detection to flag deliveries with unusual completion times or rare combinations of features.
Key visualizations were generated to assess model performance, analyze delivery patterns, and highlight detected anomalies.

# Key Findings
- **Data Preprocessing**: Calculated delivery duration, extracted time-of-day and day-of-week features, and computed Haversine distance. No missing values in key features.
- **Predictive Model (Linear Regression)**:
  - Mean Absolute Error (MAE): 147.32
  - Mean Squared Error (MSE): 260200.43
- **Anomaly Detection (Isolation Forest)**: ~5% of deliveries flagged as anomalies.
- **Characteristics of Anomalies**:
  - Extreme delivery durations (very short or long) relative to predicted values.
  - Represent rare combinations of feature values (distance, time of day, location).
- **Model Interpretation**:
  - `time_of_day` has the strongest influence (negative coefficient — later times linked to shorter durations).
  - Geographical coordinates also impact predictions.
  - Distance has negligible linear impact.
- **Visualizations**: Distribution plots, scatter plots of delivery duration vs distance/time, and interactive Plotly figure showing actual vs predicted times with anomalies highlighted.

# Instructions to Run
1. Ensure Python 3.6+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
