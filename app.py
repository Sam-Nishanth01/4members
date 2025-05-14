from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin if needed

# Load model and encoders
model = joblib.load('xgb_startup_model.pkl')
le_category = joblib.load('category_encoder.pkl')
le_status = joblib.load('status_encoder.pkl')
feature_names = ['funding_total_usd', 'category_code', 'company_age']

@app.route('/')
def index():
    categories = [
        "advertising", "analytics", "biotech", "automotive", "cleantech",
        "consulting", "ecommerce", "enterprise", "fashion", "games_video",
        "hardware", "healthhospitality", "manufacturing", "medical",
        "messaging", "mobile", "music", "network_hosting", "news", "other",
        "photo_video", "public_relations", "real_estate", "search",
        "security", "semiconductor", "social", "software", "transportation", "travel", "web"
    ]
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        funding = float(data['funding_total_usd'])
        category = data['category_code'].lower()
        age = float(data['company_age'])

        if category not in le_category.classes_:
            return jsonify({'error': f"Unknown category '{category}'"}), 400

        category_encoded = le_category.transform([category])[0]
        X_input = np.array([[funding, category_encoded, age]])

        proba = model.predict_proba(X_input)[0][1]
        prediction = int(proba >= 0.5)
        status = le_status.inverse_transform([prediction])[0]

        return jsonify({
    'prediction': str(status),
    'probability_of_acquisition': float(round(proba, 4))
})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
