from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load('/home/4Members/startup_predictor/xgb_startup_model.pkl')
le_category = joblib.load('/home/4Members/startup_predictor/category_encoder.pkl')
le_status = joblib.load('/home/4Members/startup_predictor/status_encoder.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_acquisition():
    try:
        # Read from form
        funding = float(request.form['funding_total_usd'])
        category_code = request.form['category_code']
        age = float(request.form['company_age'])

        # Encode category
        category_encoded = le_category.transform([category_code.lower()])[0]

        # Prepare input
        X_input = np.array([[funding, category_encoded, age]])

        # Predict
        proba = model.predict_proba(X_input)[0][1]
        prediction = int(proba >= 0.5)
        status_label = le_status.inverse_transform([prediction])[0]

        return render_template('index.html',
                               prediction=status_label,
                               probability=round(proba * 100, 2))

    except Exception as e:
        return render_template('index.html', prediction="Error", probability=str(e))

if __name__ == '__main__':
    app.run(debug=True)
