import pandas
import numpy

# load data from excel file dataset
data = pandas.read_excel('AMLFinalData.xlsx')
data.columns = data.columns.str.strip()

# get the target column
targetCol = "No. of Dwellings per Person"
Acode = "SA2 Area Code"

# attach each areas 2016 ratio to its 2021 row using the sa2 code (debugged using AI)
prev = data[data["Year"] == 2016][[Acode, targetCol]].rename(columns={targetCol: "prev"})
test = data[data["Year"] == 2021].merge(prev, on=Acode, how="left")

# evaluation metrics
err = test["prev"] - test[targetCol]
print("RMSE:", numpy.sqrt((err ** 2).mean()))
print("MAE:", err.abs().mean())