import sklearn.linear_model, sklearn.model_selection, sys, common, commongbg
from common import YEAR_ST, YEAR_END, PARAMS, TOP_FIVE, np
from sklearn.metrics import mean_squared_error, mean_absolute_error

pos = sys.argv[1]
seasons, smean, sstd, names = common.allDataParse(YEAR_ST,YEAR_END, pos)
raw, mean, std, acs, mean_acc, std_acc, _ = commongbg.allDataParse(YEAR_ST,YEAR_END, pos)
nopl, years, weeks, stats = raw.shape

total_loss = 0
total_mae_loss = 0

d_slice = [names.index(i) for i in TOP_FIVE[pos]]
t5_dict = dict()

X_train = raw[:, -2, :, :].reshape(nopl * weeks, stats)
X_test = raw[:, -1, :, :].reshape(nopl * weeks, stats)

for i in range(len(PARAMS[pos])):
    Y_train = acs[:, -2, :, i].reshape(nopl * weeks,)
    Y_test = acs[:, -1, :, i].reshape(nopl * weeks,)

    lr = sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(), {'fit_intercept':[True,False], 'alpha':[0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95]}, n_jobs=-1, cv=5)

    lr.fit(X_train, Y_train)
    y_test_pred = lr.best_estimator_.predict(X_test)

    test = Y_test.reshape(nopl, weeks) * std_acc[-1, :, i] + mean_acc[-1, :, i]
    pred = y_test_pred.reshape(nopl, weeks) * std[-1, :, i] + mean[-1, :, i]

    normal_test = (np.sum(test, axis=1) - smean[-1, i])/sstd[-1, i]
    normal_pred = (np.sum(pred, axis=1) - smean[-1, i])/sstd[-1, i]

    loss = mean_squared_error(normal_test, normal_pred)
    mae_loss = mean_absolute_error(normal_test, normal_pred)

    total_loss += loss
    total_mae_loss += mae_loss

    for pl_i in range(len(d_slice)):
        pl = TOP_FIVE[pos][pl_i]
        if pl not in t5_dict:
            t5_dict[pl] = []
        pred_c = np.sum(pred[d_slice[pl_i]])
        test_c = np.sum(test[d_slice[pl_i]])
        t5_dict[pl].append((pred_c, test_c))

    print (PARAMS[pos][i], loss, mae_loss, mean_absolute_error(np.sum(test, axis=1), np.sum(pred, axis=1)))

print ('Avg', total_loss/len(PARAMS[pos]), total_mae_loss/len(PARAMS[pos]), '-')
print(t5_dict)
