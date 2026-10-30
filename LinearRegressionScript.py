import pandas
import numpy
import math # for log features
from sklearn.linear_model import LinearRegression

# load data from excel file dataset
data_train = pandas.read_excel('AMLFinalTrainingData.xlsx')
data_train.columns = data_train.columns.str.strip()
data_test = pandas.read_excel('AMLFinalTestingData.xlsx')
data_test.columns = data_test.columns.str.strip()

# log transform features (uncomment thesse if the log transform features are to be used)
# data_train["logPop"] = data_train["Population (No.)"].apply(math.log)
# data_train["logDensity"] = data_train["Population Density"].apply(math.log)
# data_train["logGrowth"] = data_train["Population Growth (5 years)"].apply(math.log1p)
# data_train["logDistance"] = data_train["Distance to CBD (km)"].apply(math.log1p)

# data_test["logPop"] = data_test["Population (No.)"].apply(math.log)
# data_test["logDensity"] = data_test["Population Density"].apply(math.log)
# data_test["logGrowth"] = data_test["Population Growth (5 years)"].apply(math.log1p)
# data_test["logDistance"] = data_test["Distance to CBD (km)"].apply(math.log1p)

# featureCol = ["GCCSA", "logDistance", "logPop", "logDensity", "logGrowth"]

# seperate the features and target
targetCol = "No. of Dwellings per Person"
featureCol = ["GCCSA", "Distance to CBD (km)", "Population (No.)", "Population Density", "Population Growth (5 years)"]
x = data_train[featureCol] # feature
y = data_train[targetCol] # target

# train the model
model = LinearRegression()
model.fit(x, y)
# print(dict(zip(featureCol, model.coef_)))

# test the model
predict = model.predict(data_test[featureCol])
# evaluation metrics
err = predict - data_test[targetCol]
print("RMSE:", numpy.sqrt((err ** 2).mean()))
print("MAE:", err.abs().mean())