import joblib
import pandas as pd

# -------------------------------------------------------
# 1. Load Saved Models & Encoders
# -------------------------------------------------------
health_model = joblib.load("soil_health_model.pkl")
fert_model = joblib.load("fertilizer_model.pkl")

soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
health_encoder = joblib.load("health_encoder.pkl")
fert_encoder = joblib.load("fert_encoder.pkl")

# -------------------------------------------------------
# 2. Define Sample Input Data
# -------------------------------------------------------
sample_input = {
    "Temperature": 28.5,
    "Humidity": 62.5,
    "Moisture": 44.0,
    "Soil_pH": 6.8,
    "Electrical_Conductivity": 1.3,
    "Organic_Carbon": 0.7,
    "Nitrogen": 230.0,
    "Phosphorus": 26.0,
    "Potassium": 165.0,
    "Sulfur": 17.5,
    "Zinc": 1.35,
    "Iron": 6.0,
    "Soil_Type": "Loamy",
    "Crop_Type": "Wheat",
    "Soil_Health_Score": 66.0,
}


def predict_soil_data(input_dict):
    # Prepare DataFrame
    data = pd.DataFrame([input_dict])

    # Transform categorical fields
    data["Soil_Type"] = soil_encoder.transform(data["Soil_Type"])
    data["Crop_Type"] = crop_encoder.transform(data["Crop_Type"])

    # Predict
    health_pred_encoded = health_model.predict(data)
    fert_pred_encoded = fert_model.predict(data)

    # Inverse Transform Predictions
    health_result = health_encoder.inverse_transform(health_pred_encoded)[0]
    fertilizer_result = fert_encoder.inverse_transform(fert_pred_encoded)[0]

    # Calculate Confidence Level
    confidence = round(
        float(health_model.predict_proba(data).max()) * 100, 2
    )

    return {
        "Soil_Health_Category": health_result,
        "Recommended_Fertilizer": fertilizer_result,
        "Confidence_Score": f"{confidence}%",
    }


# -------------------------------------------------------
# 3. Run Inference
# -------------------------------------------------------
if __name__ == "__main__":
    result = predict_soil_data(sample_input)
    print("=== Soil Prediction Results ===")
    for key, value in result.items():
        print(f"{key}: {value}")