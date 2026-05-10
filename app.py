from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

try:
    model = joblib.load("model.joblib")
    target_names = joblib.load("target_names.joblib")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

@app.route("/health", methods=["GET"])
def health():
    """Returns the service status."""
    return jsonify({"status": "healthy"}), 200

def validate_features(features):
    """Helper to validate iris feature inputs."""
    if features is None:
        return False, "Missing 'features' key in request body."
    if not isinstance(features, list):
        return False, "'features' must be a list."
    if len(features) != 4:
        return False, f"Expected 4 features, but got {len(features)}."
    if not all(isinstance(x, (int, float)) for x in features):
        return False, "All feature values must be numeric."
    return True, None

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the species for a single Iris sample."""
    data = request.get_json()
    features = data.get("features")
    
    is_valid, error_msg = validate_features(features)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    prob_dict = {target_names[i]: round(float(prob), 4) for i, prob in enumerate(probabilities)}
    
    return jsonify({
        "predicted_class": target_names[prediction],
        "probabilities": prob_dict
    }), 200

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Predicts species for multiple samples."""
    data = request.get_json()
    samples = data.get("samples")
    
    if not samples or not isinstance(samples, list):
        return jsonify({"error": "Missing 'samples' list in request body."}), 400
    
    results = []
    for i, sample in enumerate(samples):
        is_valid, error_msg = validate_features(sample)
        if not is_valid:
            return jsonify({"error": f"Sample index {i}: {error_msg}"}), 400
        
        prediction = model.predict([sample])[0]
        results.append({
            "sample_index": i,
            "predicted_class": target_names[prediction]
        })
        
    return jsonify({"predictions": results}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)