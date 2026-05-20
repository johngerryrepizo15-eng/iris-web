
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('iris_model.pkl')

# Iris flower names
flower_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    prediction = model.predict([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    result = flower_names[prediction[0]]

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

