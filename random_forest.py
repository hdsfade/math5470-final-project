from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import work
from sklearn.inspection import permutation_importance


# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = "./results/random_forest/train_result.csv"
pred_result_save_path = "./results/random_forest/pred_result.csv"
base_importance_save_path = "./results/random_forest/importance_result"
base_pred_value_save_path = "./results/random_forest/pred_value"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']

start_date = "1957-01-01"  # start date
end_date = "2020-12-31"  # end date
train_start_date = start_date
valid_start_date = "1975-01-01"
test_start_date = "1987-01-01"

params = {
    'n_estimators': [300],
    'max_depth': [3, 6],
    'max_features': [30, 50, 100],
    'random_state': [12308]
}
print('--------random forest-------')
dataset = pd.read_csv(data_path)
work(RandomForestRegressor, dataset, {}, train_result_save_path, pred_result_save_path, base_importance_save_path, base_pred_value_save_path, 
     train_start_date, valid_start_date, test_start_date, end_date, True)
print('--------random forest-------')
