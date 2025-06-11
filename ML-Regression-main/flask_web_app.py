from flask import render_template, Flask, request
import numpy as np
import joblib
import functions

app = Flask(__name__)
model = joblib.load(r'D:\ML-Regression\Boston\final_polynomial_reg_model_boston.joblib')

@app.route('/')

def home():
    return render_template('website.html')




@app.route('/predict', methods=['POST'])

def predict():

    if request.method == 'POST':
        INDUS = float(request.form['INDUS'])
        RM = float(request.form['RM'])
        TAX = float(request.form['TAX'])
        PTRATIO = float(request.form['PTRATIO'])
        LSTAT = float(request.form['LSTAT'])
        X = (INDUS, RM, TAX, PTRATIO, LSTAT)
        prediction = functions.predict(X)[0][0]
        return render_template('website.html', prediction_text = "Predicted Median Value : ${:.2f}".format(prediction*1000))
    
if __name__ == '__main__':
    app.run(debug=True)

