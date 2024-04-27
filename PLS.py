from numpy import ndarray
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from utils import work
from scipy.sparse import spmatrix


# global vars
data_path = "./dataset/full_dataset.csv"
train_result_save_path = "./results/PLS/train_result.csv"
pred_result_save_path = "./results/PLS/pred_result.csv"
base_importance_save_path = "./results/PLS/importance_result"
base_value_save_path = "./results/PLS/pred_value"
train_result_columns = ['date', 'sup paras', 'r2']
pred_result_columns = ['date', 'r2', 'mse']

start_date = "1957-01-01"  # start date
end_date = "2020-12-31"  # end date
train_start_date = start_date
valid_start_date = "1975-01-01"
test_start_date = "1987-01-01"


class CompatiblePLSRegression(PLSRegression):
    def predict(self, X: ndarray | pd.DataFrame | spmatrix, copy: bool = True) -> ndarray:
        return super().predict(X, copy).squeeze()


print('---------PLS--------')
params = {'n_components': [1, 2, 3, 4, 5]}
dataset = pd.read_csv(data_path)
PLS = CompatiblePLSRegression
work(PLS, dataset, params, train_result_save_path, pred_result_save_path, base_importance_save_path, base_value_save_path,
     train_start_date, valid_start_date, test_start_date, end_date)
print('---------PLS--------')
