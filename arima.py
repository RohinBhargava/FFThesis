from common import YEAR_ST, YEAR_END, PARAMS, TOP_FIVE, YDIFF, allDataParse, np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings, sys, matplotlib.pyplot as plt, statsmodels.api as sm, math

pos = sys.argv[1]
raw, mean, std, names = allDataParse(YEAR_ST,YEAR_END, pos)

total_loss = 0
total_mae_loss = 0

d_slice = [names.index(i) for i in TOP_FIVE[pos]]
t5_dict = dict()

for i in range(len(PARAMS[pos])):
    # X_train, X_test, Y_train, Y_test = train_test_split(raw[:, :-1, i], raw[:, -1, i], test_size=0.3, random_state=1)
    X = raw[:, :-1, i]
    Y = raw[:, -1, i]
    preds = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for j in range(len(X)):
            arima = sm.tsa.statespace.SARIMAX(X[j], order=(0,1,1), enforce_stationarity=False, enforce_invertibility=False)

            fit = arima.fit(disp=0)
            # print fit.summary()
            # print (fit.forecast())

            y_test_pred = fit.forecast()[0]

            if math.isnan(y_test_pred):
                y_test_pred = 0

            preds.append(y_test_pred)

    loss = mean_squared_error(np.array(preds), Y)
    mae_loss = mean_absolute_error(np.array(preds), Y)

    total_loss += loss
    total_mae_loss += mae_loss

    for pl_i in range(len(d_slice)):
        pl = TOP_FIVE[pos][pl_i]
        if pl not in t5_dict:
            t5_dict[pl] = []
        pred = preds[d_slice[pl_i]] * std[-1, i] + mean[-1, i]
        test = Y[d_slice[pl_i]] * std[-1, i] + mean[-1, i]
        t5_dict[pl].append((pred, test))

    print (PARAMS[pos][i], loss, mae_loss, mean_absolute_error(np.array(preds) * std[-1, i] + mean[-1, i], Y * std[-1, i] + mean[-1, i]))

print ('Avg', total_loss/len(PARAMS[pos]), total_mae_loss/len(PARAMS[pos]), '-')
print(t5_dict)

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
#     for j in range(len(PARAMS[pos])):
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
# print zip(PARAMS[pos],mse_arima[min_m])
# print mse_arima_means[min_m]

# print zip(PARAMS[pos], mse_a)
# print np.mean(mse_a)
