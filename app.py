import streamlit as st
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="AI Data Center Predictive Maintenance", layout="wide")

st.title("AI Infrastructure Failure Prediction Dashboard")
st.write("Predict server failure risk using telemetry analytics and machine learning.")

@st.cache_data
def load_data():
    with zipfile.ZipFile("archive (1).zip") as z:
        with z.open("predictive_maintenance_dataset.csv") as f:
            df = pd.read_csv(f)
    return df

df = load_data()

X = df.drop(columns=["date", "device", "failure"])
y = df["failure"]

feature_labels = {
    "metric1": "CPU Load",
    "metric2": "Memory Error Count",
    "metric3": "Network Latency",
    "metric4": "Thermal Variance",
    "metric5": "Power Cycle Count",
    "metric6": "Disk I/O Activity",
    "metric7": "Cooling System Load",
    "metric8": "Fan Speed Variance",
    "metric9": "Voltage Fluctuation"
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

@st.cache_resource
def train_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_smote, y_train_smote)
    return model

model = train_model()

pred_proba_test = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, pred_proba_test)

st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Devices", f"{df['device'].nunique():,}")
col2.metric("Failure Events", f"{df['failure'].sum():,}")
col3.metric("Failure Rate", f"{df['failure'].mean()*100:.3f}%")
col4.metric("ROC-AUC", f"{roc_auc:.3f}")

st.subheader("Manual Failure Risk Prediction")

inputs = {}
cols = st.columns(3)

for i, feature in enumerate(X.columns):
    with cols[i % 3]:
        label = feature_labels.get(feature, feature)
        
        inputs[feature] = st.number_input(
            label,
            value=float(df[feature].median())
        )

input_df = pd.DataFrame([inputs])
failure_prob = model.predict_proba(input_df)[0][1]
failure_percent = failure_prob * 100

if failure_prob < 0.2:
    maintenance_window = "90+ days"
elif failure_prob < 0.5:
    maintenance_window = "30 days"
else:
    maintenance_window = "7 days"
    
st.subheader("Predicted Failure Risk")

risk_col1, risk_col2 = st.columns(2)

with risk_col1:
    st.metric("Failure Probability", f"{failure_prob*100:.2f}%")

with risk_col2:
    st.metric("Estimated Maintenance Window", maintenance_window)
    
col1, col2 = st.columns([1, 2])

with col1:
    if failure_prob >= 0.5:
        st.error("HIGH RISK: Immediate inspection recommended.")
    elif failure_prob >= 0.2:
        st.warning("MEDIUM RISK: Monitor closely.")
    else:
        st.success("LOW RISK: Normal operating condition.")

    st.info(
        f"""
Model Confidence: ROC-AUC = {roc_auc:.3f}

Prediction generated using a Random Forest predictive maintenance model.
"""
    )
        
st.subheader("Recommended Action")

if failure_prob < 0.2:
    action = "Normal operation. Continue routine monitoring."

elif failure_prob < 0.5:
    action = "Monitor system performance closely and inspect telemetry anomalies."

else:
    action = "High failure risk detected. Schedule preventive maintenance immediately."

st.success(action)
    
with col2:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=failure_percent,
        title={"text": "Failure Risk"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 20], "color": "lightgreen"},
                {"range": [20, 50], "color": "yellow"},
                {"range": [50, 100], "color": "red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

importance["feature"] = importance["feature"].map(feature_labels)

st.write("Top telemetry indicators contributing to predicted failure risk:")
st.table(importance.head(5).reset_index(drop=True))

top_feature = importance.iloc[0]["feature"]

st.info(
    f"""
Primary Risk Driver: {top_feature}

This metric showed the strongest contribution to the model's failure prediction and should be prioritized during infrastructure diagnostics.
"""
)