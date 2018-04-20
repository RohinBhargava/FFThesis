import sklearn.linear_model
from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np

raw = allDataParse(YEAR_ST,YEAR_END)

lr = sklearn.linear_model.LinearRegression()
b = [0] * len(PARAMS)
for i in range(YDIFF):
    a = lr.fit(raw[:, i], raw[:, i + 1])
    b += np.mean(np.square(a.predict(raw[:, i]) - raw[:, i + 1]), axis=0)
print zip(PARAMS,np.sqrt(np.float64(b/YDIFF)))
print np.mean(np.sqrt(np.float64(b/YDIFF)))