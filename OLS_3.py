import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import work
from sklearn.inspection import permutation_importance


# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = "./results/OLS3/train_result.csv"
pred_result_save_path = "./results/OLS3/pred_result.csv"
base_importance_save_path = "./results/OLS3/importance_result"
base_pred_value_save_path = "./results/OLS3/pred_value"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']

start_date = "2020-01-01"  # start date
end_date = "2020-12-31"  # end date
train_start_date = start_date
valid_start_date = "2020-06-01"
test_start_date = "2020-08-01"

print('---------OLS3--------')
dataset = pd.read_csv(data_path)
features_3_target = ['mvel1', 'mom1m', 'mom6m', 'mom12m', 'mom36m', 'bm', 'DATE','RET']
dataset = dataset[features_3_target]

OLS = LinearRegression
work(OLS, dataset, {}, train_result_save_path, pred_result_save_path, base_importance_save_path, base_pred_value_save_path, 
     train_start_date, valid_start_date, test_start_date, end_date)
print('---------OLS--------')
