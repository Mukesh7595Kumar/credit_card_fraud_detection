from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import xgboost as xgb
import keras
from tensorflow.keras.models import load_model as lm

klf = '/content/drive/MyDrive/Project/models/Fraud_Detection_Model.keras'
lstm = lm(klf)

model_filepath = '/content/drive/MyDrive/Project/models/xgb_bal_smote_model.bin'
clf = xgb.Booster()
clf.load_model(model_filepath)

path = '/content/drive/MyDrive/Project/templates1'
style = '/content/drive/MyDrive/Project/static'

app = Flask(__name__, template_folder=path, static_folder=style)

@app.route('/')
def home():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
        me = request.form['message']
        ch = request.form['choice']
        message = [float(x) for x in me.split()]
        vect = np.array(message).reshape(1, -1)
        if ch == 'lstm':
            my_prediction = lstm.predict(vect)
        else:
            my_prediction = clf.predict(xgb.DMatrix(vect))
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug = True)
