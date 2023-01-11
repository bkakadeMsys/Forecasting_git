from flask import Flask, request, jsonify, session
#from flask_cors import CORS

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
import datetime
from tsprial.forecasting import *
import pickle

import warnings

warnings.warn('ignore', category=FutureWarning)

app = Flask(__name__)
#cors = CORS(app)
ALLOWED_EXTENSIONS = (['csv'])
app.secret_key = "abcdef"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0


def train(data_file, if_train):
    sales = pd.read_csv(data_file)

    sales['date'] = pd.to_datetime(sales.date, format='%Y-%m-%d')
    sales = sales.set_index('date')
    sales['month'] = sales.index.month
    sales['year'] = sales.index.year
    sales['day'] = sales.index.day
    print(len(sales))
    sales = sales.drop(['store'], 1)

    if if_train == 'True' or if_train == 'true' or if_train == 'T':
        print("MODEL IS TRAINING.....................................")
        items_list = []
        for i in range(1, 51):
            item_train = sales[sales.item == i]
            items_list.append(item_train)

            X_train = item_train.drop(columns='sales')
            y_train = item_train['sales']

            model = ForecastingChain(
                Ridge(),
                n_estimators=24 * 3,
                lags=range(1, 24 * 7 + 1),
                use_exog=True,
                accept_nan=False
            )
            model.fit(X_train, y_train)

            filename = 'trained_models/model_item_' + str(i) + '.pkl'
            pickle.dump(model, open(filename, 'wb'))
    last_date = sales.index[-1] + datetime.timedelta(days=1)
    return last_date


def test(last_date, type_of_data, number_of):

    if type_of_data == 'W':
        totalday = number_of * 7
    if type_of_data == 'M':
        totalday = number_of * 30

    test_date = datetime.datetime.strptime(last_date.strftime('%Y-%m-%d'), '%Y-%m-%d')

    date_generated = pd.date_range(test_date, periods=totalday)

    final_test_data = pd.DataFrame()

    for i in range(1, 51):
        item_test = pd.DataFrame()
        item_test['date'] = date_generated
        item_test = item_test.set_index('date')
        item_test['item'] = i
        item_test['month'] = item_test.index.month
        item_test['year'] = item_test.index.year
        item_test['day'] = item_test.index.day

        filename = 'trained_models/model_item_' + str(i) + '.pkl'

        final_test_data['day'] = item_test.day
        final_test_data['month'] = item_test.month
        final_test_data['year'] = item_test.year

        loaded_model = pickle.load(open(filename, 'rb'))

        final_test_data['item' + str(i)] = np.round_(loaded_model.predict(item_test))

    final_test_data = final_test_data.drop(columns=['day', 'month', 'year'])
    final_test_data = final_test_data.resample(type_of_data).sum()
    print(final_test_data.head())
    final_test_data.to_csv("data/Output.csv")
    out = pd.read_csv('data/Output.csv')
    out = out.set_index('date')
    return out.to_dict('index')


@app.route('/forecasting', methods=["GET", "POST"])
def data_load():
    if request.method == 'POST':
        UPLOAD_FOLDER_1 = 'data/'
        UPLOAD_FOLDER_2 = 'trained_models/'

        type_of_data = request.form['W_or_M']
        number_of = request.form['Nos']
        if_train = str(request.form['Train'])
        print("train flag ______________", if_train)
        if os.path.isdir(UPLOAD_FOLDER_1):
            shutil.rmtree('data')
        if not os.path.isdir(UPLOAD_FOLDER_1):
            os.mkdir(UPLOAD_FOLDER_1)
        if if_train == 'True' or if_train == 'true' or if_train == 'T':
            print("REMOVE OLD MODELS........................")
            if os.path.isdir(UPLOAD_FOLDER_2):
                shutil.rmtree('trained_models')
            if not os.path.isdir(UPLOAD_FOLDER_2):
                os.mkdir(UPLOAD_FOLDER_2)

        app.config['UPLOAD_FOLDER_1'] = UPLOAD_FOLDER_1
        if 'file' not in request.files:
            return jsonify('No file part')
        file = request.files['file']
        if file.filename == '':
            return jsonify('No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER_1'], file.filename)
            file.save(filepath)
        else:
            return jsonify('Unsupported File Format')

        last_date = train(filepath, if_train)
        out = test(last_date, str(type_of_data), int(number_of))

        return out


if __name__ == '__main__':
    #app.run(debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
