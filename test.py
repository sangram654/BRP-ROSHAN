import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load data
data = pd.read_csv("Dataset.csv")
data.replace("?", np.nan, inplace=True)

# Convert numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='ignore')

data['dteday'] = pd.to_datetime(data['dteday'], errors='coerce')
data['day'] = data['dteday'].dt.day

# Cyclic encoding
data['hr_sin'] = np.sin(2 * np.pi * data['hr'] / 24)
data['hr_cos'] = np.cos(2 * np.pi * data['hr'] / 24)

data['mnth_sin'] = np.sin(2 * np.pi * data['mnth'] / 12)
data['mnth_cos'] = np.cos(2 * np.pi * data['mnth'] / 12)

data['weekday_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
data['weekday_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)

# Keep only simple features (IMPORTANT 🔥)
features = [
    'hr','mnth','weekday',
    'temp','hum','windspeed',
    'hr_sin','hr_cos',
    'mnth_sin','mnth_cos',
    'weekday_sin','weekday_cos'
]

X = data[features]
y = data['cnt']

X = X.fillna(X.median())

model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42
)

model.fit(X, y)

# Save BOTH model + feature names
joblib.dump({
    "model": model,
    "features": features
}, "bike_model.pkl", compress=3)

print("✅ Clean 12-feature model saved.")
