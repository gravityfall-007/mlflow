import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import autosklearn.regression

import dagshub
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    # Load the dataset
    file_path =( '/home/krishna/global-building-morphology-indicators/berlin_building.csv'
    )
    try:
        df = pd.read_csv(file_path, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Fill NaN values with 0 in specified columns
    columns_to_fill_with_zero = ['height', 'building:levels', 'ratio_height_to_footprint_area', 'floor_area', 'wall_area', 'envelope_area', 'volume']
    df[columns_to_fill_with_zero] = df[columns_to_fill_with_zero].fillna(0)

    # Drop specified columns
    #columns_to_remove = ['cell_country', 'cell_admin_div1', 'cell_admin_div2', 'cell_admin_div3', 'cell_admin_div4', 'cell_country_official_name', 'cell_population']
    #df.drop(columns=columns_to_remove, inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    categorical_columns = ['is_residential']
    for column in categorical_columns:
        df[column] = label_encoder.fit_transform(df[column])

    # Split the dataset into features (X) and target variable (y)
    X = df.drop(columns=['height'])
    y = df['height']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Define the AutoSklearnRegressor
    automl_model = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120, per_run_time_limit=30, n_jobs=-1)

    with mlflow.start_run():
        try:
            # Fit the AutoML model
            automl_model.fit(X_train, y_train)

            # Get predictions on the test set
            y_pred = automl_model.predict(X_test)

            # Evaluate the performance of AutoML model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print("Mean Squared Error (AutoML):", mse)
            print("R-squared (AutoML):", r2)

        except Exception as e:
            print(f"An error occurred: {e}")

        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mse)

                
        # For Remote server only (DAGShub)
        REPO_NAME = "https://github.com/gravityfall-007/mlflow.git"
        #dagshub.init(username=DAGSHUB_USERNAME, repo_name=REPO_NAME, api_token=DAGSHUB_API_TOKEN)

        remote_server_uri = "https://dagshub.com/gravityfall-007/mlflow.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                automl_model, "model", registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(automl_model, "model")

