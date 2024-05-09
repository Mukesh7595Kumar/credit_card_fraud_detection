from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import xgboost as xgb
from tensorflow.keras.models import load_model

klf = 'C:/Users/srini/PycharmProjects/pythonProject/models/Fraud_Detection_Model.keras'
lstm = load_model(klf)

model_filepath = 'C:/Users/srini/PycharmProjects/pythonProject/models/xgb_bal_smote_model.bin'
clf = xgb.Booster()
clf.load_model(model_filepath)

path = 'C:/Users/srini/PycharmProjects/pythonProject/templates'
style = 'C:/Users/srini/PycharmProjects/pythonProject/static'
upath = 'C:/Users/srini/PycharmProjects/pythonProject/uploads/'

app = Flask(__name__, template_folder=path, static_folder=style)
global_df = None


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global global_df
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'
    else:
        ul = 'C:/Users/srini/PycharmProjects/pythonProject/uploads/'
        file.save(ul + file.filename)
        global_df = pd.read_excel(upath + file.filename)
        global_df.set_index('Transaction_ID', inplace=True)
        return redirect(url_for('t_id'))


@app.route('/t_id')
def t_id():
    return render_template('t_id.html')


@app.route('/predict', methods=['POST'])
def predict():
    global global_df
    if request.method == 'POST':
        tid = request.form['message']
        ch = request.form['choice']
        row = global_df.loc[tid]
        message = [float(x) for x in row.values]
        vect = np.array(message).reshape(1, -1)
        if ch == 'lstm':
            my_prediction = lstm.predict(vect)
        else:
            my_prediction = clf.predict(xgb.DMatrix(vect))
        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
