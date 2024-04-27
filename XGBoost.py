import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import work
from xgboost import XGBRegressor

# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = "./results/XGBoost/train_result.csv"
pred_result_save_path = "./results/XGBoost/pred_result.csv"
base_importance_save_path = "./results/XGBoost/importance_result"
base_value_save_path = "./results/XGBoost/pred_value"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']

tart_date = "1957-01-01"  # start date
end_date = "2020-12-31"  # end date
train_start_date = start_date
valid_start_date = "1975-01-01"
test_start_date = "1987-01-01"

sup_paras = {
    'n_estimators': [500, 800, 1000],
    'max_depth': [1, 2],
    'random_state': [10086],
    'learning_rate': [.1]
}

print('---------XGBoost--------')
dataset = pd.read_csv(data_path)
work(XGBRegressor, dataset, sup_paras, train_result_save_path, pred_result_save_path, base_importance_save_path, base_value_save_path,
     train_start_date, valid_start_date, test_start_date, end_date)
print('---------XGBoost--------')
