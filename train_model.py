import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------------
# 1. Configuration & Data Loading
# -------------------------------------------------------
DATASET_FILE = "soil_dataset.csv"

if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"'{DATASET_FILE}' not found. Place it in the same directory."
    )

df = pd.read_csv(DATASET_FILE)
print(f"Dataset loaded successfully with {len(df)} records.\n")

# Features expected by models
FEATURE_COLS = [
    "Temperature",
    "Humidity",
    "Moisture",
    "Soil_pH",
    "Electrical_Conductivity",
    "Organic_Carbon",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Sulfur",
    "Zinc",
    "Iron",
    "Soil_Type",
    "Crop_Type",
    "Soil_Health_Score",
]

# -------------------------------------------------------
# 2. Data Preprocessing & Categorical Encoding
# -------------------------------------------------------
X = df[FEATURE_COLS].copy()

# Initialize Encoders
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
health_encoder = LabelEncoder()
fert_encoder = LabelEncoder()

# Fit and Transform Inputs
X["Soil_Type"] = soil_encoder.fit_transform(X["Soil_Type"])
X["Crop_Type"] = crop_encoder.fit_transform(X["Crop_Type"])

# Fit and Transform Targets
y_health = health_encoder.fit_transform(df["Soil_Health_Category"])
y_fert = fert_encoder.fit_transform(df["Recommended_Fertilizer"])

# Train-Test Split (80% Train / 20% Test)
X_train, X_test, yh_train, yh_test, yf_train, yf_test = train_test_split(
    X, y_health, y_fert, test_size=0.2, random_state=42
)

# -------------------------------------------------------
# 3. Model Training
# -------------------------------------------------------
print("Training Soil Health Model...")
health_model = RandomForestClassifier(n_estimators=100, random_state=42)
health_model.fit(X_train, yh_train)

print("Training Fertilizer Recommendation Model...")
fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
fert_model.fit(X_train, yf_train)

# -------------------------------------------------------
# 4. Evaluation
# -------------------------------------------------------
yh_pred = health_model.predict(X_test)
yf_pred = fert_model.predict(X_test)

print(
    f"Soil Health Model Test Accuracy: {accuracy_score(yh_test, yh_pred) * 100:.2f}%"
)
print(
    f"Fertilizer Model Test Accuracy:  {accuracy_score(yf_test, yf_pred) * 100:.2f}%\n"
)

# -------------------------------------------------------
# 5. Save Models and Artifacts
# -------------------------------------------------------
joblib.dump(health_model, "soil_health_model.pkl")
joblib.dump(fert_model, "fertilizer_model.pkl")
joblib.dump(soil_encoder, "soil_encoder.pkl")
joblib.dump(crop_encoder, "crop_encoder.pkl")
joblib.dump(health_encoder, "health_encoder.pkl")
joblib.dump(fert_encoder, "fert_encoder.pkl")

print("All models and encoders saved successfully to disk!")