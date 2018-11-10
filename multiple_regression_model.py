from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
# Load the data from the boston house-prices dataset
boston_data = load_boston()
x = boston_data['data']
y = boston_data['target']
