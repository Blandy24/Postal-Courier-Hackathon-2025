# 📦 PostaConnect - AI-powered Delivery Intelligence System

## (a) Team Name & Members

**Team Name:** Postal Hackathon 2025
**Members:** Blandina Kakore

---

## (b) Problem

Efficient last-mile delivery is a critical challenge in Tanzania's postal and courier sector. Delays, inconsistent delivery times, and untracked anomalies in deliveries negatively impact customer satisfaction. The goal is to leverage AI and data analytics to forecast delivery durations, detect anomalous deliveries, and provide actionable insights to improve operations.

---

## (c) Solution Overview

PostaConnect is a Streamlit-based web application that uses machine learning to predict delivery durations and detect anomalies in real-time. The solution includes:

* **Delivery Duration Prediction:** A Linear Regression model predicts delivery times using factors such as pickup time, day of the week, distance, and geo-coordinates.
* **Anomaly Detection:** An Isolation Forest identifies deliveries that deviate significantly from normal patterns.
* **Visual Analytics:** Interactive dashboards display delivery distributions, model predictions, anomaly locations on a Tanzanian map, and allow users to download anomaly data.
* **Scalability:** The solution is designed to work with both uploaded datasets and a default sample dataset for demonstration.

This approach enables postal operators to proactively address delayed or suspicious deliveries, optimize routes, and monitor operational efficiency.

---

## (d) Instructions to Run the Project

### **Requirements**

* Python 3.10+
* Git (for version control and deployment)

### **Steps**

1. Clone the repository:

```bash
git clone https://github.com/Blandy24/Postal-Courier-Hackathon-2025.git
cd Postal-Courier-Hackathon-2025
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app locally:

```bash
streamlit run app.py
```

5. Upload your dataset if needed, or use the default sample dataset.

---

## (e) Special Requirements

**Libraries:**

* `pandas`, `numpy`
* `scikit-learn`
* `haversine`
* `streamlit`
* `plotly`
* `matplotlib`
* `joblib`

**Datasets:**

* Default: `DATA/sample_delivery.csv` (for demo purposes)
* Optional: Upload your own CSV dataset with the following required columns:

```
order_id, poi_lat, poi_lng, receipt_lat, receipt_lng, sign_time, receipt_time
```

**APIs:** None required (all processing is local).


# Dashboard / Visualization Output

* **Web-based App**: [https://postaconnect.streamlit.app/](https://postaconnect.streamlit.app/)
* **Features**:

  * Overview of delivery data
  * Model predictions vs actuals
  * Anomaly detection and interactive map
  * Downloadable anomalies CSV
