import sklearn.linear_model, sys
from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

raw = allDataParse(YEAR_ST,YEAR_END, sys.argv[1])
total_loss = 0

for i in range(len(PARAMS)):
    X_train, X_test, Y_train, Y_test = train_test_split(raw[:, :-1, i], raw[:, -1, i], test_size=0.3, random_state=1)

    lr = sklearn.linear_model.LinearRegression()

    lr.fit(X_train, Y_train)
    y_test_pred = lr.predict(X_test)

    loss = mean_squared_error(Y_test, y_test_pred)

    total_loss += loss

    print PARAMS[i], loss

print total_loss/len(PARAMS)

# b = [0] * len(PARAMS)
# for i in range(YDIFF):
#     a = lr.fit(raw[:, i], raw[:, i + 1])
#     b += mean_squared_error(raw[:, i + 1], a.predict(raw[:, i]), multioutput='raw_values')
# # a = lr.fit(raw[:, -2], raw[:, -1])
# # b += mean_squared_error(raw[:, -1], a.predict(raw[:, -2]), multioutput='raw_values')
# print zip(PARAMS,b/YDIFF)
# print np.mean(b/YDIFF)