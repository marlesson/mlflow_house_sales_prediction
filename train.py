# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
sns.set(style="white")

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def plot_residuo(actual, pred):
    g = sns.jointplot(x=actual, y=pred, kind="reg", color="m", height=7)
    g.savefig('artefacts/img_res_0.png')

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

    max_depth    = int(sys.argv[1])
    n_estimators = int(sys.argv[2])

    with mlflow.start_run():
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        
        # train
        model.fit(train_x, train_y)

        # evaluate
        predicted_qualities = model.predict(test_x)
        (rmse, mae, r2)     = eval_metrics(test_y, predicted_qualities)

        plot_residuo(test_y, predicted_qualities)

        print("RandomForestRegressor model (max_depth=%d, n_estimators=%d):" % (max_depth, n_estimators))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE",  mae)
        mlflow.log_metric("R2",   r2)
        mlflow.log_artifacts("artefacts/")

        mlflow.sklearn.log_model(model, "model")
