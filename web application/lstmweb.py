from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

# load the model from disk
file = 'C:/Users/srini/PycharmProjects/pythonProject/models/Fraud_Detection_Model.keras'
clf = load_model(file)

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
		print(type(me))
		message = [float(x) for x in me.split()]
		vect = np.array(message).reshape(1, -1)
		print("Input data:", vect)
		my_prediction = clf.predict(vect)
		print("Prediction result:", my_prediction)
	return render_template('result.html',prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
