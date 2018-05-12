import sklearn.linear_model, sys, common, commongbg
from common import YEAR_ST, YEAR_END, PARAMS, np
from sklearn.metrics import mean_squared_error

pos = sys.argv[1]
raw, _, _, acs, _, _, _ = commongbg.allDataParse(YEAR_ST,YEAR_END, pos)
seasons, mean, std, names = common.allDataParse(YEAR_ST,YEAR_END, pos)

total_loss = 0

print mean_squared_error(seasons[:, -2, :], seasons[:, -1, :])

X_train = raw[:, -2, :, :].reshape(len(raw) * 17, len(raw[0][0][0]))
X_test = raw[:, -1, :, :].reshape(len(raw) * 17, len(raw[0][0][0]))

for i in range(len(PARAMS[pos])):
    Y_train = acs[:, -2, :, i].reshape(2822, 1)
    Y_test = acs[:, -1, :, i].reshape(2822, 1)

    lr = sklearn.linear_model.LinearRegression()


    lr.fit(X_train, Y_train)
    y_test_pred = lr.predict(X_test)

    loss = mean_squared_error(np.mean(Y_test.reshape(166,17), axis=1), np.mean(y_test_pred.reshape(166,17), axis=1))

    total_loss += loss

    print (PARAMS[pos][i], loss, np.sqrt(loss * std[-1, i] + mean[-1, i]),  mean[-1, i], std[-1, i])


print (total_loss/len(PARAMS[pos]))