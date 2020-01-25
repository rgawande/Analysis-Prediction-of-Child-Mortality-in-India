import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import pickle
from randomforest import sm

from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/')
def Home():
   return render_template('template.html')

@app.route('/form')
def SocioEconomic():
   return render_template('SocioEconomic.html')

@app.route('/form2')
def disease():
   return render_template('diseases.html')

@app.route('/results',methods = ['POST', 'GET'])
def results():
    if request.method == 'POST':
        a1 = request.form['BirthRate']
        a11=float(a1)
        a2 = request.form['IMR']
        a12=float(a2)
        a3 = request.form['BLP']
        a13=float(a3)
        a4 = request.form['Underfive']
        a14=float(a4)
        a5 = request.form['Fertility']
        a15=float(a5)
        a6 = request.form['MotherLit']
        a16=float(a6)
        a7 = request.form['Priedu']
        a17=float(a7)
        a8 = request.form['Undernourishment']
        a18=float(a8)
        a9 = request.form['Water']
        a19=float(a9)
        a10 = request.form['Sanitation']
        a110=float(a10)
        a=[2018,129999999,a12,a13,a14,a15,a16,a17,a18,a19,3.6,1082993]
       
        dataset = pd.DataFrame(a) 
        dataset = dataset.transpose()
        X = dataset.iloc[:, 2:10].values
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        labelencoder = LabelEncoder()
        X[:, 3] = labelencoder.fit_transform(X[:, 3])
        onehotencoder = OneHotEncoder(categorical_features = [3])
        X = onehotencoder.fit_transform(X).toarray()
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        X_test = sc.fit_transform(X)  
        reg = pickle.loads(sm)
        y_pred=reg.predict(X_test)
        
        
        return render_template('results.html',a1=a1,a2=a2,a3=a3,a4=a4,a5=a5,a6=a6,a7=a7,a8=a8,a9=a9,a10=a10,y_pred=y_pred)

if __name__ == '__main__':
    app.run(debug = True)
    
    
 