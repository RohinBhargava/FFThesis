import sklearn.linear_model, sys, common, commongbg
from common import YEAR_ST, YEAR_END, PARAMS, np
from sklearn.metrics import mean_squared_error

pos = sys.argv[1]
raw, mean, std, acs, mean_acc, std_acc, names = commongbg.allDataParse(YEAR_ST,YEAR_END, pos)
seasons, _, _, _ = common.allDataParse(YEAR_ST,YEAR_END, pos)

total_loss = 0

for i in range(len(PARAMS[pos])):
    X_train = raw[:, -2, :, i]
    X_test = raw[:, -1, :, i]
    Y_train = acs[:, -2, :, i]
    Y_test = seasons[:, -1, i]

    lr = sklearn.linear_model.LinearRegression()

    lr.fit(X_train, Y_train)
    y_test_pred = lr.predict(X_test)

    print Y_test - np.sum(y_test_pred, axis=1)/17
    loss = mean_squared_error(Y_test, np.sum(y_test_pred, axis=1)/17)

    total_loss += loss

    print (PARAMS[pos][i], loss)


print (total_loss/len(PARAMS[pos]))