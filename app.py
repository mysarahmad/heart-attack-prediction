from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('heart_attack.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index1.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        chest_pain_type = request.form['chest_pain_type']
        Exercise_Include_Angina = request.form['Exercise_Include_Angina']
        oldpeak = request.form['oldpeak']
        caa = request.form['caa']
        thall = request.form['thall']


        values = np.array([[chest_pain_type,Exercise_Include_Angina,oldpeak,caa,thall]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)