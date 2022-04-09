from statistics import mode
from flask import Flask, render_template ,jsonify
import pickle
import numpy as np
import joblib
import pandas as pd

from flask import request
app = Flask(__name__,template_folder='templates')


def predict(values):
    print("hel")
    if len(values) == 5:
        model = joblib.load('./model/ckd.pkl')
        print(model)
        values = np.asarray(values)
        print("values",values)
        print(values.shape)
        #df = pd.DataFrame(values.reshape(-1,len(values)), columns = ['age','bp','sg','al','su','rbc','pc','pcc','ba','bgr', 'bu', 'sc', 'sod', 'pot','hemo','pcv','wbcc','rbcc','htn','dm','cad','appet','pe','ane'])
        df = pd.DataFrame(values.reshape(-1,len(values)), columns = ['bgr','bu','sc','pcv','wbcc'])
        #print(df[0:1])
        val = model.predict(df)
        print("val",val[0])
        val = val[0]
        return val

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
            print(to_predict_dict)
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print(to_predict_list)
            pred = predict(to_predict_list)
            return render_template("pred.html", pred = pred)

    except:
        pred = "Please enter valid Data"
        return render_template("pred.html", pred = pred)

    return render_template('prediction.html')
    

if __name__ == '__main__':
    app.run(debug=True)


 