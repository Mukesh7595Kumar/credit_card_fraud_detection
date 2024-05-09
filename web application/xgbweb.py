from flask import Flask, render_template, request
import numpy as np
import xgboost as xgb

model_filepath = 'C:/Users/srini/PycharmProjects/pythonProject/models/xgb_bal_smote_model.bin'
clf = xgb.Booster()
clf.load_model(model_filepath)
path = 'C:/Users/srini/PycharmProjects/pythonProject/templates'
style = 'C:/Users/srini/PycharmProjects/pythonProject/static'

app = Flask(__name__, template_folder=path, static_folder=style)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        me = request.form['message']
        message = [float(x) for x in me.split()]
        vect = np.array(message).reshape(1, -1)
        my_prediction = clf.predict(xgb.DMatrix(vect))
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)


