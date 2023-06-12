from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST','GET'])
def predict():
    data = pd.read_csv('CR_DataN.csv')
    y = data[['Crop', 'Diseases']]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y.values)
    features=[float(x) for x in request.form.values()]
    model=pickle.load(open('model.pkl','rb'))
    output= model.predict([features])
    output_data = mlb.inverse_transform(output)
    print(output_data)
    print(len(output_data))
    return render_template('index.html', crop=output_data[0][0],disease=output_data[0][1])

if __name__ == '__main__':
    app.run(debug=True)

