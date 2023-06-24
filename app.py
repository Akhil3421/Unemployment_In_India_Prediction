import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd



app=Flask(__name__)
model=pickle.load(open('rf.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))
encoderr=pickle.load(open('encoder.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data=request.json['data']
    print(data)
    new_data=stdd(data)
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

def stdd(data):
    column_names = ['Region','Date','Estimated Employed','Estimated Labour Participation Rate (%)','Area']
    df = pd.DataFrame([data], columns=column_names)
    nums = df.drop(columns=['Region','Area'])
    nums['Date']=nums['Date'].str.replace('-','')
    nums= nums.astype(float)
    new_cat=pd.DataFrame(encoderr.transform([[df['Region'][0],df['Area'][0]]]))
    new_cat= new_cat.astype(float)
    new_nums=scaler.transform(nums)
    new_data=pd.concat([new_cat,pd.DataFrame(new_nums)],axis=1)
    return new_data


@app.route('/predict',methods=['POST'])
def predict():
    #data = [request.form[x] for x in ['Date', 'IsHoliday', 'Type', 'Store', 'Dept', 'Size', 'Temperature', 'Fuel_Price', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']]    
    data=[x for x in request.form.values()]
    final_input=stdd(data)
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Unemployment rate is {}".format(output))



if __name__=="__main__":
    app.run()