import joblib
import pandas as pd

# Load model and encoders once
model = joblib.load('/home/4Members/startup_predictor/xgb_startup_model.pkl')
category_encoder = joblib.load('/home/4Members/startup_predictor/category_encoder.pkl')
status_encoder = joblib.load('/home/4Members/startup_predictor/status_encoder.pkl')

def predict_acquisition(startup_dict, threshold=0.5):
    df = pd.DataFrame([startup_dict])

    # Ensure all required columns exist
    required_columns = model.get_booster().feature_names
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    # Encode categorical features
    df['category_code'] = category_encoder.transform(df['category_code'])

    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= threshold)
    label = status_encoder.inverse_transform([pred])[0]

    return {
        'probability_acquired': round(prob, 3),
        'predicted_status': label
    }
