from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model, symptom_list, y_encoded = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv('disease_dataset.csv')
le = LabelEncoder()
le.fit(df['prognosis'])

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        selected = request.form.getlist('symptoms')
        input_data = [1 if symptom in selected else 0 for symptom in symptom_list]
        pred = model.predict([input_data])[0]
        prediction = le.inverse_transform([pred])[0]
    return render_template('index.html', symptoms=symptom_list, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
