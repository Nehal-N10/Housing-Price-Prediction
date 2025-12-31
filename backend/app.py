 
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5500"])

# Load artifacts
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
training_features = pd.read_csv("training_features.csv")['feature_names'].tolist()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Create DataFrame with expected raw features
        input_df = pd.DataFrame([[
            data.get("MSZoning", "RL"),
            data.get("LotFrontage", 0),
            data.get("LotArea", 0),
            data.get("LandContour", "Lvl"),
            data.get("Neighborhood", "NAmes"),
            data.get("BldgType", "1Fam"),
            data.get("HouseStyle", "1Story"),
            data.get("OverallCond", 5),
            data.get("YearBuilt", 2000),
            data.get("YearRemodAdd", 2000),
            data.get("Foundation", "PConc"),
            data.get("Heating", "GasA"),
            data.get("CentralAir", "Y"),
            data.get("Electrical", "SBrkr"),
            data.get("FullBath", 2),
            data.get("HalfBath", 1),
            data.get("BedroomAbvGr", 3),
            data.get("KitchenAbvGr", 1),
            data.get("Fireplaces", 0),
            data.get("GarageType", "Attchd"),
            data.get("GarageYrBlt", 2000),
            data.get("GarageCars", 2),
            data.get("PoolArea", 0),
            data.get("YrSold", 2020)
        ]], columns=[
            "MSZoning", "LotFrontage", "LotArea", "LandContour", "Neighborhood",
            "BldgType", "HouseStyle", "OverallCond", "YearBuilt", "YearRemodAdd",
            "Foundation", "Heating", "CentralAir", "Electrical", "FullBath",
            "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "Fireplaces", "GarageType",
            "GarageYrBlt", "GarageCars", "PoolArea", "YrSold"
        ])

        # Create dummy variables
        input_encoded = pd.get_dummies(input_df, drop_first=True)
        
        # Align with training features
        missing_cols = set(training_features) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
            
        input_encoded = input_encoded[training_features]
        
        # Convert to numeric
        input_encoded = input_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Scale features
        scaled_data = scaler.transform(input_encoded)
        
        # Predict and convert to Python float
        prediction = model.predict(scaled_data)[0]
        return jsonify({"predicted_price": round(float(prediction), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 

if __name__ == "__main__":
    app.run(debug=True)