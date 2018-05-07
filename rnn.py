# https://jasdeep06.github.io/posts/Understanding-LSTM-in-Tensorflow-MNIST/
from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import rnn
import warnings, tensorflow as tf, sys

raw = allDataParse(YEAR_ST,YEAR_END, sys.argv[1])
raw = tup[0]
for i in range(len(tup[0])):
    raw[i] -= tup[1]
    for j in range(len(tup[2])):
        for k in range(len(tup[2][j])):
            if tup[2][j][k] != 0:
                raw[i][j][k] /= tup[2][j][k]
            else:
                raw[i][j][k] = 0

total_loss = 0
for i in range(len(PARAMS)):
    X_train = raw[:, :-2, i]
    X_test = raw[:, 1:-1, i]
    Y_train = raw[:, -2, i]
    Y_test = raw[:, -1, i]

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, YDIFF - 1, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

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
    train_step = tf.train.RMSPropOptimizer(0.1).minimize(cost)

    sess.run(tf.global_variables_initializer())

    accuracyl = []
    for b in range(10):
        train_step.run(feed_dict={x: X_train.reshape(len(X_train), YDIFF - 1, 1), y_: Y_train.reshape(len(Y_train), 1)})
        accuracy = tf.losses.mean_squared_error(y_, y)
        accuracyl.append(accuracy.eval(feed_dict={x: X_test.reshape(len(X_test), YDIFF - 1, 1), y_: Y_test.reshape(len(Y_test), 1)}))

    loss = min(accuracyl)
    total_loss += loss
    print (PARAMS[i], loss)
print (total_loss/len(PARAMS))
