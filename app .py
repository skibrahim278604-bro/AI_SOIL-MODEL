import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# -------------------------------------------------------
# Load or Train Machine Learning Models from soil_dataset.csv
# -------------------------------------------------------

DATASET_PATH = "soil_dataset.csv"

# Categorical encoders
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()

# Feature columns matching soil_dataset.csv
FEATURE_COLS = [
    "Temperature", "Humidity", "Moisture", "Soil_pH",
    "Electrical_Conductivity", "Organic_Carbon", "Nitrogen",
    "Phosphorus", "Potassium", "Sulfur", "Zinc", "Iron",
    "Soil_Type", "Crop_Type", "Soil_Health_Score"
]

def load_or_train_models():
    """Load pre-trained models or dynamically train from soil_dataset.csv."""
    global health_model, fert_model, soil_encoder, crop_encoder, fert_amount_map

    if os.path.exists("soil_health_model.pkl") and os.path.exists("fertilizer_model.pkl"):
        health_model = joblib.load("soil_health_model.pkl")
        fert_model = joblib.load("fertilizer_model.pkl")
        soil_encoder = joblib.load("soil_encoder.pkl")
        crop_encoder = joblib.load("crop_encoder.pkl")
    else:
        print("Training models on soil_dataset.csv...")
        df = pd.read_csv(DATASET_PATH)

        # Prepare dataset features
        X = df[FEATURE_COLS].copy()
        X["Soil_Type"] = soil_encoder.fit_transform(X["Soil_Type"])
        X["Crop_Type"] = crop_encoder.fit_transform(X["Crop_Type"])

        # Train Soil Health Category model
        health_model = RandomForestClassifier(n_estimators=50, random_state=42)
        health_model.fit(X, df["Soil_Health_Category"])

        # Train Recommended Fertilizer model
        fert_model = RandomForestClassifier(n_estimators=50, random_state=42)
        fert_model.fit(X, df["Recommended_Fertilizer"])

        # Save trained artifacts
        joblib.dump(health_model, "soil_health_model.pkl")
        joblib.dump(fert_model, "fertilizer_model.pkl")
        joblib.dump(soil_encoder, "soil_encoder.pkl")
        joblib.dump(crop_encoder, "crop_encoder.pkl")

    # Map average dosage (kg/acre) per fertilizer from dataset
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        fert_amount_map = df.groupby("Recommended_Fertilizer")["Fertilizer_Amount_kg_per_acre"].mean().to_dict()
    else:
        fert_amount_map = {
            "Compost": 500, "DAP": 100, "MOP": 150, 
            "NPK 10-26-26": 150, "NPK 19-19-19": 100, 
            "Urea": 150, "Vermicompost": 150
        }

load_or_train_models()

# -------------------------------------------------------
# Home Route
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        soil_types=sorted(list(soil_encoder.classes_)),
        crop_types=sorted(list(crop_encoder.classes_)),
        result=None
    )

# -------------------------------------------------------
# Prediction Route
# -------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Parse numeric inputs from request.form
        temperature = float(request.form["Temperature"])
        humidity = float(request.form["Humidity"])
        moisture = float(request.form["Moisture"])
        soil_ph = float(request.form["Soil_pH"])
        ec = float(request.form["Electrical_Conductivity"])
        organic_carbon = float(request.form["Organic_Carbon"])
        soil_health_score = float(request.form["Soil_Health_Score"])

        nitrogen = float(request.form["Nitrogen"])
        phosphorus = float(request.form["Phosphorus"])
        potassium = float(request.form["Potassium"])
        sulfur = float(request.form["Sulfur"])
        zinc = float(request.form["Zinc"])
        iron = float(request.form["Iron"])

        soil_type_str = request.form["Soil_Type"]
        crop_type_str = request.form["Crop_Type"]

        # 2. Encode categorical variables
        soil_encoded = soil_encoder.transform([soil_type_str])[0]
        crop_encoded = crop_encoder.transform([crop_type_str])[0]

        # 3. Create DataFrame matching model expectations
        input_data = pd.DataFrame([{
            "Temperature": temperature,
            "Humidity": humidity,
            "Moisture": moisture,
            "Soil_pH": soil_ph,
            "Electrical_Conductivity": ec,
            "Organic_Carbon": organic_carbon,
            "Nitrogen": nitrogen,
            "Phosphorus": phosphorus,
            "Potassium": potassium,
            "Sulfur": sulfur,
            "Zinc": zinc,
            "Iron": iron,
            "Soil_Type": soil_encoded,
            "Crop_Type": crop_encoded,
            "Soil_Health_Score": soil_health_score
        }])

        # 4. Generate Predictions
        health_result = health_model.predict(input_data)[0]
        fertilizer_result = fert_model.predict(input_data)[0]

        # 5. Calculate Confidence Level
        probs = health_model.predict_proba(input_data)
        confidence = round(float(probs.max()) * 100, 2)

        # 6. Dosage & Recommendations
        recommended_amount = int(fert_amount_map.get(fertilizer_result, 150))

        if fertilizer_result in ["Compost", "Vermicompost"]:
            recommendation = f"Apply {recommended_amount} kg/acre of {fertilizer_result} to boost organic matter and improve long-term soil structure."
        elif fertilizer_result == "Urea":
            recommendation = f"Nitrogen levels are deficient. Apply {recommended_amount} kg/acre of Urea in split applications."
        elif fertilizer_result in ["DAP", "NPK 19-19-19"]:
            recommendation = f"Correct phosphorus/balanced nutrient needs by applying {recommended_amount} kg/acre of {fertilizer_result} during sowing."
        elif fertilizer_result in ["MOP", "NPK 10-26-26"]:
            recommendation = f"Address potassium deficiency by applying {recommended_amount} kg/acre of {fertilizer_result}."
        else:
            recommendation = f"Apply {recommended_amount} kg/acre of {fertilizer_result} according to standard agricultural guidelines."

        result = {
            "soil_health": health_result,
            "fertilizer": fertilizer_result,
            "amount": recommended_amount,
            "confidence": confidence,
            "recommendation": recommendation
        }

        return render_template(
            "index.html",
            soil_types=sorted(list(soil_encoder.classes_)),
            crop_types=sorted(list(crop_encoder.classes_)),
            result=result
        )

    except Exception as e:
        return render_template(
            "index.html",
            soil_types=sorted(list(soil_encoder.classes_)),
            crop_types=sorted(list(crop_encoder.classes_)),
            error=str(e),
            result=None
        )

# -------------------------------------------------------
# Run Application
# -------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)