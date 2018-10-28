import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sys import stdin
import warnings
warnings.filterwarnings('ignore')

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def train():
    print("please input your training data file address:")
    print("for example:","C:/Users/Administrator/Desktop/volumental/trainset.csv")

    filename = stdin.readline()
    filename = filename.strip('\n')

    df = pd.read_csv(filename,index_col=0)

    # delete non valid data
    counts = df['length_size'].value_counts()
    va = list(counts[counts == 1].index)
    df = df[(~df['length_size'].isin(va))]

    counts = df['width_size'].value_counts()
    va = list(counts[counts == 1].index)
    df = df[(~df['width_size'].isin(va))]

    #numberize the shoe style
    df['shoe_style'] = df['shoe_style'].where(df['shoe_style'] != 'a', 0)
    df['shoe_style'] = df['shoe_style'].where(df['shoe_style'] != 'm', 1)

    #generate dependent and independent variables
    y_train = df[['length_size', 'width_size']]
    X_train = df.drop(['length_size', 'width_size'], axis=1)

    #train the model
    max_depth = 30
    rfr = RandomForestRegressor(n_estimators=100, max_depth=max_depth,random_state=2)
    rfr.fit(X_train, y_train)

    #save the model into file
    model =  'randomforestmodel.pkl'
    joblib.dump(rfr, model)

    print()
    print("model is saved as: "+ model)
    print()

    unique_length = y_train.length_size.unique()
    unique_width = y_train.width_size.unique()

    return unique_length, unique_width

def predict():
    print("please input your test data file address:")
    print("for example: C:/Users/Administrator/Desktop/volumental/testset.csv")
    filename = stdin.readline()
    filename = filename.strip('\n')

    #reading the test file
    X_test = pd.read_csv(filename, index_col=0)

    #load the model
    print()
    print("please input your model file address:")
    print("for example: randomforestmodel.pkl")
    model = stdin.readline()
    model = model.strip('\n')
    rfr = joblib.load(model)

    #prepare the test data
    X_test['shoe_style'] = X_test['shoe_style'].where(X_test['shoe_style'] != 'a', 0)
    X_test['shoe_style'] = X_test['shoe_style'].where(X_test['shoe_style'] != 'm', 1)

    #predict by the model
    y_rf = rfr.predict(X_test)

    #Y_test.read_csv("C:/Users/Administrator/Desktop/volumental/Y_test.csv")
    #rfr.score(X_test, Y_test)

    #prepare output
    p_l = []
    p_w = []
    for i in y_rf:
        p_l.append(find_nearest(unique_length, i[0]))
        p_w.append(find_nearest(unique_width, i[1]))

    test = X_test.copy()
    test['length_size'] = p_l
    test['width_size'] = p_w

    #write the output
    path = os.path.dirname(filename)
    resultfile = path+"/predict.csv"
    test.to_csv(resultfile)
    print()
    print("the result is written at", path+"/", "as 'predict.csv'")

if __name__== "__main__":
    unique_length, unique_width = train()
    predict()