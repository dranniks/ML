# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Загрузка данных
home_data = pd.read_csv('D:\Projects\AI\ML\dtr\melb_data.csv')

# Целевая переменная
y = home_data.Price

# Выбор признаков (исключаем 'Address')
features = ['Rooms', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt']
X = home_data[features]

# Удаление строк с пропущенными значениями
X_clean = X.dropna()
y_clean = y[X_clean.index]

# Разделение данных
train_X, val_X, train_y, val_y = train_test_split(X_clean, y_clean, random_state=1)

# Создание и обучение модели
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# Предсказание
predictions = model.predict(val_X)
print("Первые 10 предсказаний:")
print(predictions[:10], '\n')
print("Первые 10 исходных значений:")
print(val_y[:10], '\n')

# Make validation predictions and calculate mean absolute error
# val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

def mean_absolute_percentage_error(y_pred, y_true): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

candidate_max_leaf_nodes = [i for i in range(5, 500)]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)
print("Best max leaf nodes: ", best_tree_size, '\n')

# Using best value for max_leaf_nodes
model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
model.fit(train_X, train_y)
predictions = model.predict(val_X)
val_mae = mean_absolute_error(predictions, val_y)
val_mape = mean_absolute_percentage_error(predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae), '\n')
print("Validation MAPE for best value of max_leaf_nodes: {:,.0f}".format(val_mape), '\n')

print("Первые 10 предсказаний:")
print(predictions[:10], '\n')
