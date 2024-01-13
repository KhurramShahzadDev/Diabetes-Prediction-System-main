

from flask import Flask, render_template, request

import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder='template')

# Update the model path to the correct one
model_path = 'C:/Users/Bravo_is_Back/Downloads/Diabetes-Prediction-System-main/saved_models/diabetes_model.h5'
model = load_model(model_path)

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['blood_pressure'])
        skin_thickness = float(request.form['skin_thickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['diabetes_pedigree_function'])
        age = float(request.form['age'])

        new_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
        prediction = model.predict(np.array(new_data))
        outcome = "diabetic-1" if prediction > 0.5 else "not diabetic-0"

        return render_template('result.html', outcome=outcome)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
