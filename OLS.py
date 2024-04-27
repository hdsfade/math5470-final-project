import pandas as pd
from sklearn.linear_model import LinearRegression
from utils import work


# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = "./results/OLS/train_result.csv"
pred_result_save_path = "./results/OLS/pred_result.csv"
base_importance_save_path = "./results/OLS/importance_result"
base_value_save_path = "./results/OLS/pred_value"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']

start_date = "1957-01-01"  # start date
end_date = "2020-12-31"  # end date
train_start_date = start_date
valid_start_date = "1975-01-01"
test_start_date = "1987-01-01"

print('---------OLS--------')
dataset = pd.read_csv(data_path)
OLS = LinearRegression
work(OLS, dataset, {}, train_result_save_path, pred_result_save_path, base_importance_save_path, base_value_save_path, 
     train_start_date, valid_start_date, test_start_date, end_date)
print('---------OLS--------')
