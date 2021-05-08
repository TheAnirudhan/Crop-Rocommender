import pickle
import numpy as np
from flask import Flask, render_template, request

filename = 'crop-recommendation-knn-model.pkl'
knn = pickle.load(open(filename, 'rb'))

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        n = float(request.form['Nitrogen_Level'])
        p = float(request.form['Phosphorus_Level'])
        k = float(request.form['Potassium_Level'])
        t = float(request.form['Temperature'])
        h = float(request.form['Humidity'])
        ph =float(request.form['pH'])
        r = float(request.form['Rainfall'])

        data = np.array([[n, p, k, t, h, ph, r]])
        prediction = knn.predict(data)
        return render_template('prediction_page.html',crop = prediction[0].capitalize())
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
