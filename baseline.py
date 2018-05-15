import sklearn.linear_model, sys
from common import YEAR_ST, YEAR_END, PARAMS, TOP_FIVE, allDataParse, np
from sklearn.metrics import mean_squared_error, mean_absolute_error

pos = sys.argv[1]
raw, mean, std, names = allDataParse(YEAR_ST,YEAR_END, pos)

total_loss = 0
total_mae_loss = 0

d_slice = [names.index(i) for i in TOP_FIVE[pos]]
t5_dict = dict()

for i in range(len(PARAMS[pos])):
    X_train = raw[:, :-2, i]
    X_test = raw[:, 1:-1, i]
    Y_train = raw[:, -2, i]
    Y_test = raw[:, -1, i]

    lr = sklearn.linear_model.LinearRegression()

    lr.fit(X_train, Y_train)
    y_test_pred = lr.predict(X_test)

    loss = mean_squared_error(Y_test, y_test_pred)
    mae_loss = mean_absolute_error(y_test_pred, Y_test)

    total_loss += loss
    total_mae_loss += mae_loss

    for pl_i in range(len(d_slice)):
        pl = TOP_FIVE[pos][pl_i]
        if pl not in t5_dict:
            t5_dict[pl] = []
        pred = y_test_pred[d_slice[pl_i]] * std[-1, i] + mean[-1, i]
        test = Y_test[d_slice[pl_i]] * std[-1, i] + mean[-1, i]
        t5_dict[pl].append((pred, test))

    print (PARAMS[pos][i], loss, mae_loss, mean_absolute_error(y_test_pred * std[-1, i] + mean[-1, i], Y_test * std[-1, i] + mean[-1, i]))

print ('Avg', total_loss/len(PARAMS[pos]), total_mae_loss/len(PARAMS[pos]), '-')
print(t5_dict)

# b = [0] * len(PARAMS[pos])
# for i in range(YDIFF):
#     a = lr.fit(raw[:, i], raw[:, i + 1])
#     b += mean_squared_error(raw[:, i + 1], a.predict(raw[:, i]), multioutput='raw_values')
# # a = lr.fit(raw[:, -2], raw[:, -1])
# # b += mean_squared_error(raw[:, -1], a.predict(raw[:, -2]), multioutput='raw_values')
# print zip(PARAMS[pos],b/YDIFF)
# print np.mean(b/YDIFF)
