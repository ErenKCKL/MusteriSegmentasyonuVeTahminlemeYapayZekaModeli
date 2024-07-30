from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

kmeans_model = joblib.load('kmeans_model.pkl')
regression_model = joblib.load('regression_model.pkl')
classification_model = joblib.load('classification_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/kmeans', methods=['POST'])
def kmeans_predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = kmeans_model.predict(features)
    return jsonify({'cluster': int(prediction[0])})

@app.route('/regression', methods=['POST'])
def regression_predict():
    data = request.get_json(force=True)
    annual_income = np.array(data['annual_income']).reshape(-1, 1)
    prediction = regression_model.predict(annual_income)
    return jsonify({'spending_score': float(prediction[0])})

@app.route('/classification', methods=['POST'])
def classification_predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = classification_model.predict(scaled_features)
    return jsonify({'spending_class': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)