from statistics import mode
from src.utils.all_utils import read_yaml, save_report, create_dir
import argparse
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def evaluate_metrics(actual_value, predicted_value):
    rmse = np.sqrt(mean_squared_error(actual_value, predicted_value))
    mae = mean_absolute_error(actual_value, predicted_value)
    r2 = r2_score(actual_value, predicted_value)
    return rmse, mae, r2


def evaaluate(config_path):
    config = read_yaml(config_path)

    artifacts_dir = config['artifacts']['artifacts_dir']
    split_data_dir = config['artifacts']['split_data_dir']
    test_data_filename = config['artifacts']['test']

    test_data_path = os.path.join(artifacts_dir, split_data_dir,
                                  test_data_filename)
    test_data = pd.read_csv(test_data_path)

    test_y = test_data['quality']
    test_x = test_data.drop(columns=['quality'])

    model_dir = config['artifacts']['model_dir']
    model_filename = config['artifacts']['model_file']
    model_path = os.path.join(artifacts_dir, model_dir, model_filename)

    lr = joblib.load(model_path)

    predicted_value = lr.predict(test_x)
    rmse, mae, r2 = evaluate_metrics(test_y, predicted_value)
    print(rmse, mae, r2)

    scores_dir = config['artifacts']['report_dir']
    scores_file = config['artifacts']['scores']
    scores_dir_path = os.path.join(artifacts_dir, scores_dir)
    create_dir([scores_dir_path])

    scores_file_path = os.path.join(scores_dir_path, scores_file)

    scores = {'rmse': rmse, 'mae': mae, 'r2': r2}
    save_report(scores, report_path=scores_file_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', "-c", default='config/config.yaml')

    parsed_args = args.parse_args()

    evaaluate(config_path=parsed_args.config)
