from statistics import mode
from flask import Flask, render_template ,jsonify
import pickle
import numpy as np
import joblib
import pandas as pd

from flask import request
app = Flask(__name__,template_folder='templates')


def predict(values):
    if len(values) == 5:
        model = joblib.load('./model/ckd.pkl')
        scaler = joblib.load('./model/scaler.bin')
       # print(model)
        values = np.asarray(values)
        #print("values",values)
        #print(values.shape)
        values = pd.DataFrame(values.reshape(-1,5), columns = ['hemo', 'pcv', 'sc', 'sg', 'rbcc'])
        #print(df[0:1])
        values = scaler.transform(values)
        df = pd.DataFrame(values.reshape(-1,5), columns = ['hemo', 'pcv', 'sc', 'sg', 'rbcc'])
       # print(values)
        val = model.predict(df)
        #print("val",val)
        val = val[0]
        return val
#17.10	41.0	0.80	1.020	5.2
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/prediction', methods = ['POST', 'GET'])
def predict_ckd():
    pred = ""
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            #print(to_predict_dict)
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            #print(to_predict_list)
            pred = predict(to_predict_list)
            #print(pred)
            return render_template("pred.html", pred = pred)

    except:
        pred = "Please enter valid Data"
        return render_template("pred.html", pred = pred)

    return render_template('prediction.html')
    

if __name__ == '__main__':
    app.run(debug=True)


 