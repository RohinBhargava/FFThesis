import sklearn.linear_model, sklearn.model_selection, sys, common, commongbg
from common import YEAR_ST, YEAR_END, PARAMS, np
from sklearn.metrics import mean_squared_error, mean_absolute_error

pos = sys.argv[1]
seasons, smean, sstd, names = common.allDataParse(YEAR_ST,YEAR_END, pos)
raw, mean, std, acs, mean_acc, std_acc, _ = commongbg.allDataParse(YEAR_ST,YEAR_END, pos)
nopl, years, weeks, stats = raw.shape

total_loss = 0

X_train = raw[:, -2, :, :].reshape(nopl * weeks, stats)
X_test = raw[:, -1, :, :].reshape(nopl * weeks, stats)

for i in range(len(PARAMS[pos])):
    Y_train = acs[:, -2, :, i].reshape(nopl * weeks,)
    Y_test = acs[:, -1, :, i].reshape(nopl * weeks,)

    lr = sklearn.model_selection.GridSearchCV(sklearn.linear_model.LinearRegression(), {'fit_intercept':[True,False]}, n_jobs=-1, cv=5)

    lr.fit(X_train, Y_train)
    y_test_pred = lr.best_estimator_.predict(X_test)

    test = Y_test.reshape(nopl, weeks) * std_acc[-1, :, i] + mean_acc[-1, :, i]
    pred = y_test_pred.reshape(nopl, weeks) * std[-1, :, i] + mean[-1, :, i]

    normal_test = (np.sum(test, axis=1) - smean[-1, i])/sstd[-1, i]
    normal_pred = (np.sum(pred, axis=1) - smean[-1, i])/sstd[-1, i]

    loss = mean_squared_error(normal_test, normal_pred)

    total_loss += loss

    print (lr.best_params_, PARAMS[pos][i], loss, mean_absolute_error(normal_pred, normal_test) * sstd[-1, i])

print (total_loss/len(PARAMS[pos]))
