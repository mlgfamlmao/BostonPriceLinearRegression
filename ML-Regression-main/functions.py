import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def predict(l):
    model = joblib.load(filename=r'D:\ML-Regression\Boston\final_polynomial_reg_model_boston.joblib')


    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    l = poly_features.fit_transform([l,])

    y = model.predict(l)
    return y

print(predict([1,2,3,4,5])[0][0])
