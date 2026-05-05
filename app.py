from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load models
cervical_model = pickle.load(open("cervical_model.pkl", "rb"))
breast_model = pickle.load(open("breast_model.pkl", "rb"))

@app.route("/")
def home():
    return "OncoPredict API Running"

@app.route("/predict_cervical", methods=["POST"])
def predict_cervical():
    data = request.json["features"]
    prediction = cervical_model.predict([data])
    return jsonify({"prediction": int(prediction[0])})

@app.route("/predict_breast", methods=["POST"])
def predict_breast():
    data = request.json["features"]
    prediction = breast_model.predict([data])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)