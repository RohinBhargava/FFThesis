# https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, TOP_FIVE, allDataParse, np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.contrib import rnn
import warnings, tensorflow as tf, sys

pos = sys.argv[1]
raw, mean, std, names = allDataParse(YEAR_ST,YEAR_END, pos)

d_slice = [names.index(i) for i in TOP_FIVE[pos]]
t5_dict = dict()

total_loss = 0
for i in range(len(PARAMS[pos])):
    X_train = raw[:, :-2, i]
    X_test = raw[:, 1:-1, i]
    Y_train = raw[:, -2, i]
    Y_test = raw[:, -1, i]

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, YDIFF - 1, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def weight_variable(shape):
      initial = tf.contrib.layers.xavier_initializer(uniform=True,seed=3)
      return tf.Variable(initial(shape))

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def RNN(x, weights, biases):
        x = tf.unstack(x, YDIFF - 1, 1)
        lstm_cell = rnn.BasicLSTMCell(len(raw), forget_bias=1.0)
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], weights) + biases

    W_out = weight_variable([len(raw), 1])
    b_out = bias_variable([1])

    y = RNN(x, W_out, b_out)
    cost = tf.losses.mean_squared_error(y_, y)
    train_step = tf.train.AdamOptimizer(0.02).minimize(cost)

    tests = None
    preds = None

    accuracy = tf.losses.mean_squared_error(y_, y)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    loss = float('inf')
    ave_loss = None
    for b in range(50):
        fd = {x: X_train.reshape(len(X_train), YDIFF - 1, 1), y_: Y_train.reshape(len(Y_train), 1)}
        fd_test = {x: X_test.reshape(len(X_test), YDIFF - 1, 1), y_: Y_test.reshape(len(Y_test), 1)}
        train_step.run(feed_dict=fd)
        r = accuracy.eval(feed_dict=fd)
        if r < loss:
            loss = min(loss, r)
            tests = Y_test
            preds = y.eval(feed_dict=fd)

    for pl_i in range(len(d_slice)):
        pl = TOP_FIVE[pos][pl_i]
        if pl not in t5_dict:
            t5_dict[pl] = []
        pred = float(preds[d_slice[pl_i]] * std[-1, i] + mean[-1, i])
        test = float(tests[d_slice[pl_i]] * std[-1, i] + mean[-1, i])
        t5_dict[pl].append((pred, test))

    sess.close()

    total_loss += loss
    print (PARAMS[pos][i], loss, mean_absolute_error(preds, tests) * std[-1, i])

print (total_loss/len(PARAMS[pos]), t5_dict)
