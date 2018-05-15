import sklearn.linear_model, sys, common, commongbg
from common import YEAR_ST, YEAR_END, PARAMS, TOP_FIVE, np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.contrib import rnn
import warnings, tensorflow as tf, sys

pos = sys.argv[1]
seasons, smean, sstd, names = common.allDataParse(YEAR_ST,YEAR_END, pos)
raw, mean, std, acs, mean_acc, std_acc, _ = commongbg.allDataParse(YEAR_ST,YEAR_END, pos)
nopl, years, weeks, stats = raw.shape

total_loss = 0

d_slice = [names.index(i) for i in TOP_FIVE[pos]]
t5_dict = dict()

X_train = raw[:, -2, :, :]
X_test = raw[:, -1, :, :]

for i in range(len(PARAMS[pos])):
    Y_train = acs[:, -2, :, i]
    Y_test = acs[:, -1, :, i]

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, weeks, stats])
    y_ = tf.placeholder(tf.float32, shape=[None, weeks])

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def RNN(x, weights, biases):
        x = tf.unstack(x, weeks,  1)
        lstm_cell = rnn.BasicLSTMCell(len(raw), forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights) + biases

    W_out = weight_variable([len(raw), weeks])
    b_out = bias_variable([weeks])

    y = RNN(x, W_out, b_out)
    cost = tf.losses.mean_squared_error(y_, y)
    train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

    test = y_ * std_acc[-1, :, i] + mean_acc[-1, :, i]
    pred = y * std[-1, :, i] + mean[-1, :, i]

    tests = None
    preds = None

    accuracy = tf.losses.mean_squared_error((tf.reduce_sum(test, 1) - smean[-1, i])/sstd[-1, i], (tf.reduce_sum(pred, 1) - smean[-1, i])/sstd[-1, i])

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    loss = float('inf')
    ave_loss = None
    for b in range(50):
        fd = {x: X_train.reshape(nopl, weeks, stats), y_: Y_train.reshape(nopl,  weeks)}
        fd_test = {x: X_test.reshape(nopl, weeks, stats), y_: Y_test.reshape(nopl,  weeks)}
        train_step.run(feed_dict=fd)
        r = accuracy.eval(feed_dict=fd_test)
        if r < loss:
            loss = min(loss, r)
            tests = Y_test * std_acc[-1, :, i] + mean_acc[-1, :, i]
            preds = pred.eval(feed_dict=fd_test)

    sess.close()
    total_loss += loss

    for pl_i in range(len(d_slice)):
        pl = TOP_FIVE[pos][pl_i]
        if pl not in t5_dict:
            t5_dict[pl] = []
        pred_c = np.sum(preds[d_slice[pl_i]])
        test_c = np.sum(tests[d_slice[pl_i]])
        t5_dict[pl].append((pred_c, test_c))

    print (PARAMS[pos][i], loss, mean_absolute_error(np.sum(tests, axis=1), np.sum(preds, axis=1)))

print (total_loss/len(PARAMS[pos]), t5_dict)
