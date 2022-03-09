import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('./models/lin_reg_model_v1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    car_age=int(request.form.get("car_age"))
    km=int(request.form.get("car_age"))
    final_features = np.array([np.array([km,car_age])])
    car_price = model.predict(final_features)

    output = round(car_price[0], 2)

    return render_template('index.html', prediction_text='Car price should be  {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)