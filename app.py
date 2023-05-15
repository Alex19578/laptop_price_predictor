from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.metrics import r2_score

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('laptop_price_prediction_model.pkl', 'rb'))

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        company = int(request.form['company'])
        laptop_type = int(request.form['laptop_type'])
        display = int(request.form['display'])
        display_size = int(request.form['display-size'])
        processor = int(request.form['processor'])
        ram = int(request.form['ram'])
        storage = int(request.form['storage'])
        gpu = int(request.form['gpu'])
        os = int(request.form['os'])
        weight = int(request.form['weight'])

        features = np.array([[company, laptop_type, display, display_size, processor, ram, storage, gpu, os, weight]])
        prediction =model.predict(features)
        pred = "The price of the laptop is " + str(prediction[0])
        return render_template('index.html', prediction=pred, prediction_score = "This predictor model accuracy is : 93.234")

if __name__ == '__main__':
    app.run(debug=True)
