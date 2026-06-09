# AI Infrastructure Failure Prediction Dashboard

A machine learning dashboard for predicting server failure risk using telemetry analytics.

## Features

- Random Forest failure prediction model
- SMOTE class balancing
- Interactive risk prediction dashboard
- Feature importance analysis
- Maintenance recommendation engine
- Failure risk gauge visualization

## Model Performance

- ROC-AUC: 0.851

### ROC Curve
<img width="419" height="357" alt="Screenshot 2026-06-09 at 3 01 32 PM" src="https://github.com/user-attachments/assets/a92332b9-6ac0-482c-8359-a5bff36e4e09" />

### Confusion Matrix
<img width="407" height="293" alt="Screenshot 2026-06-09 at 3 02 24 PM" src="https://github.com/user-attachments/assets/55934c72-1711-44eb-9fca-d1eb43c2b802" />

### Feature Importance
<img width="579" height="356" alt="Screenshot 2026-06-09 at 3 02 55 PM" src="https://github.com/user-attachments/assets/2bb1995a-989f-47ac-98c6-878361b0f28f" />

## Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas
- Plotly

## Note

Dataset not included due to size.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py

