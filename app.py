import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import model2
import statistics

app = Flask(__name__ )

model = pickle.load(open("model.pkl","rb"))

def standardized_input_value(a,b):
    return (b-statistics.mean(a)) / (statistics.pstdev(a))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predict():
    get_preg = request.form.get("Pregnancies")
    get_glucose = request.form.get("Glucose")
    get_bp = request.form.get("BloodPressure")
    get_bmi = request.form.get("BMI")
    get_pedigree = request.form.get("DiabetesPedigreeFunction")
    get_age = request.form.get("Age")
    get_insulin = request.form.get("Insulin")

    features = [float(get_preg) , float(get_glucose), float(get_bp), float(get_insulin), float(get_bmi), float(get_pedigree), float(get_age)]

    #features = [float(x) for x in request.form.values()]
    #features = [np.array(float_features)]

    model2.preg.append(features[0])
    model2.glucose.append(features[1])
    model2.blood_pressure.append(features[2])
    model2.insulin.append(features[3])
    model2.bmi.append(features[4])
    model2.pedigree.append(features[5])
    model2.age.append(features[6])

    inp_preg = standardized_input_value(model2.preg, features[0])
    inp_glucose = standardized_input_value(model2.glucose, features[1])
    inp_bp = standardized_input_value(model2.blood_pressure, features[2])
    inp_insulin = standardized_input_value(model2.insulin, features[3])
    inp_bmi = standardized_input_value(model2.bmi, features[4])
    inp_pedigree = standardized_input_value(model2.pedigree, features[5])
    inp_age = standardized_input_value(model2.age, features[6])

    final_features = [float(inp_preg),float(inp_glucose),float(inp_bp),float(inp_insulin),float(inp_bmi),float(inp_pedigree),float(inp_age) ]
    final_features2 = [np.array(final_features)]

    prediction = model.predict(final_features2)


    if prediction == 1:
        return render_template("index.html",prediction_text = "SORRY YOU HAVE DIABETES CONSULT A DOCTOR IMMEDIATELY")
    else:
        return render_template("index.html",prediction_text = "NICE YOU DONT HAVE DIABETES")


if __name__ == "__main__":
    app.run(debug=True)
