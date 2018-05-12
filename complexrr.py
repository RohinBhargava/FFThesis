import sklearn.linear_model, sys, common, commongbg
from common import YEAR_ST, YEAR_END, PARAMS, np
from sklearn.metrics import mean_squared_error

pos = sys.argv[1]
raw, _, _, acs, _, _, _ = commongbg.allDataParse(YEAR_ST,YEAR_END, pos)
seasons, mean, std, names = common.allDataParse(YEAR_ST,YEAR_END, pos)
nopl, years, weeks, stats = raw.shape

total_loss = 0

X_train = raw[:, -2, :, :].reshape(nopl * weeks, stats)
X_test = raw[:, -1, :, :].reshape(nopl * weeks, stats)

for i in range(len(PARAMS[pos])):
    Y_train = acs[:, -2, :, i].reshape(nopl * weeks,)
    Y_test = acs[:, -1, :, i].reshape(nopl * weeks,)

    lr = sklearn.linear_model.Ridge(alpha=0.5)

    lr.fit(X_train, Y_train)
    y_test_pred = lr.predict(X_test)

    loss = mean_squared_error(np.mean(Y_test.reshape(nopl, weeks), axis=1), np.mean(y_test_pred.reshape(nopl, weeks), axis=1))

    total_loss += loss

    print (PARAMS[pos][i], loss, np.sqrt(loss) * std[-1, i])


print (total_loss/len(PARAMS[pos]))
