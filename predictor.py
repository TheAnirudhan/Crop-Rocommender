import pickle
import numpy as np
from flask import Flask, render_template, request

filename = 'crop-recommendation-knn-model.pkl'
knn = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        n = int(request.form['Nitrogen_Level'])
        p = int(request.form['Phosphorus_Level'])
        k = int(request.form['Potassium_Level'])
        t = int(request.form['Temperature'])
        h = int(request.form['Humidity'])
        ph = int(request.form['pH'])
        r = int(request.form['Rainfall'])

        data = np.array([[n, p, k, t, h, ph, r]])
        prediction = knn.predict(data)
        return render_template('prediction_page.html',crop = prediction[0].capitalize())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
