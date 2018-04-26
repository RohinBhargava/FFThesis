import sklearn.linear_model
from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np
from sklearn.metrics import mean_squared_error

raw = allDataParse(YEAR_ST,YEAR_END)

lr = sklearn.linear_model.LinearRegression()
b = [0] * len(PARAMS)
for i in range(YDIFF):
    a = lr.fit(raw[:, i], raw[:, i + 1])
    b += mean_squared_error(raw[:, i + 1], a.predict(raw[:, i]), multioutput='raw_values')
print zip(PARAMS,b)
print np.mean(b)