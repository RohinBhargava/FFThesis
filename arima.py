from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
import warnings, sys, matplotlib.pyplot as plt, statsmodels.api as sm




tup = allDataParse(YEAR_ST,YEAR_END, sys.argv[1])
raw = tup[0]

total_loss = 0

for i in range(len(PARAMS)):
    # X_train, X_test, Y_train, Y_test = train_test_split(raw[:, :-1, i], raw[:, -1, i], test_size=0.3, random_state=1)
    X = raw[:, :-1, i]
    Y = raw[:, -1, i]
    preds = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        h = 0
        for j in range(len(X)):
            # arima = ARIMA(X[j], (1, 1, 2))
            arima = sm.tsa.statespace.SARIMAX(X[j], order=(1,1,1), enforce_stationarity=False, enforce_invertibility=False)

            fit = arima.fit(disp=False)
            # print fit.summary()
            # print (fit.forecast())

            y_test_pred = fit.forecast()[0]

            preds.append(y_test_pred)

            if abs(y_test_pred - Y[h]) > 20:
                plt.plot(X[j])
                plt.show()
            h += 1

    loss = mean_squared_error(Y, np.array(preds))
    total_loss += loss

    print (PARAMS[i], loss)

print (total_loss/len(PARAMS))

# mse_arima = []
# mse_arima_means = []
# m = []

# for p in range(4):
#     for q in range(5):
#         for d in range(5):
#             try:
# arimas = []
# actuals = []
# for i in raw:
#     a = []
#     b = []
#     for j in range(len(PARAMS)):
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore")
#             a.append(ARIMA(i[:-1, j], (0,1,0)).fit(disp=0).forecast()[0][0])
#             b.append(i[-1, j])
#     arimas.append(a)
#     actuals.append(b)
#
# mse_a = mean_squared_error(np.array(actuals), np.array(arimas), multioutput='raw_values')
                # m.append((p, q, d))
                # mse_arima.append(mse_a)
                # mse_arima_means.append(np.mean(mse_a))
            # except:
            #     pass

# min_m = mse_arima_means.index(min(mse_arima_means))
# print m[min_m]
# print zip(PARAMS,mse_arima[min_m])
# print mse_arima_means[min_m]

# print zip(PARAMS, mse_a)
# print np.mean(mse_a)
