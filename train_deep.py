# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
import seaborn as sns
sns.set(style="white")

import mlflow
import mlflow.keras


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def plot_residuo(actual, pred):
    g = sns.jointplot(x=actual, y=pred, kind="reg", color="m", height=7)
    g.savefig('artefacts/img_res_0.png')

def plot_hist(hist):
    # summarize history for loss
    fig, ax = plt.subplots()  # create figure & 1 axis

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    return fig 

def build_model(input_shape, layers):
  # Model
  
  model = Sequential()
  model.add(Dense(layers[0], input_shape=input_shape))
  model.add(Activation('relu'))
  
  for i in layers[1:]:
    model.add(Dense(i))
    model.add(Activation('relu'))

  model.add(Dense(1, activation='linear'))

  model.compile(optimizer='rmsprop', loss='mse')
  return model

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/raw.csv")
    data = pd.read_csv(path, index_col=['id'])

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "price"
    train_x = train.drop(["date","price"], axis=1)
    test_x  = test.drop(["date","price"], axis=1)
    train_y = train["price"]
    test_y  = test["price"]
    
    #params
    layers    = eval(sys.argv[1])
    epochs    = int(sys.argv[2])

    with mlflow.start_run():
      #model
        model = build_model((train_x.shape[1],), layers)

        # train
        hist = model.fit(train_x, train_y, 
                validation_data=(test_x, test_y), 
                batch_size=254, epochs=epochs)

        # evaluate
        predicted_qualities = model.predict(test_x).reshape(-1)
        (rmse, mae, r2)     = eval_metrics(test_y, predicted_qualities)

        plot_hist(hist).savefig('artefacts/train_hist.png')
        plot_residuo(test_y, predicted_qualities)

        print(model.summary())
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("layers", layers)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE",  mae)
        mlflow.log_metric("R2",   r2)
        mlflow.log_artifacts("artefacts/")

        mlflow.keras.log_model(model, "model")
