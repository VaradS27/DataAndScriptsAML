import pandas
import numpy
from xgboost import DMatrix, train

# load data from excel file dataset
data_train = pandas.read_excel('AMLFinalTrainingData.xlsx')
data_train.columns = data_train.columns.str.strip()
data_test = pandas.read_excel('AMLFinalTestingData.xlsx')
data_test.columns = data_test.columns.str.strip()

# separate the features and target
targetCol = "No. of Dwellings per Person"
featureCol = ["GCCSA", "Distance to CBD (km)", "Population (No.)", 
              "Population Density", "Population Growth (5 years)"]
x = data_train[featureCol] # feature
y = data_train[targetCol] # target

# creating DMatrix from x and y
dmatrix = DMatrix(data=x, label=y)
# set XGBoost parameters
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.1,
    'random_state': 42,
    'max_depth': 3
}

# train the model
model = train(params, dmatrix, num_boost_round=300)
print(model.get_score(importance_type="gain"))

# test the model
predict = model.predict(DMatrix(data=data_test[featureCol]))
err = predict - data_test[targetCol]
print("RMSE:", numpy.sqrt((err ** 2).mean()))
print("MAE:", err.abs().mean())
