from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load models
cervical_model = pickle.load(open("cervical_model.pkl", "rb"))
breast_model = pickle.load(open("breast_model.pkl", "rb"))

@app.route("/")
def home():
    return "OncoPredict API Running"

@app.route("/predict_cervical", methods=["POST"])
def predict_cervical():

    try:
        data = request.json["features"]

        features = np.array(data).reshape(1, -1)

        prediction = cervical_model.predict(features)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

@app.route("/predict_breast", methods=["POST"])
def predict_breast():

    try:
        data = request.json["features"]

        features = np.array(data).reshape(1, -1)

        prediction = breast_model.predict(features)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
