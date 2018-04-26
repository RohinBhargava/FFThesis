from common import YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np
from sklearn.metrics import mean_squared_error
from tensorflow.contrib import rnn
import warnings, tensorflow as tf

raw = allDataParse(YEAR_ST,YEAR_END)
print raw.shape

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, len(PARAMS)])
y_ = tf.placeholder(tf.float32, shape=[len(PARAMS)])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def RNN(x, weights, biases):
    x = tf.unstack(x, x.shape[0], 1)
    lstm_cell = rnn.BasicLSTMCell(len(PARAMS), forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    print outputs[-1].shape
    return tf.matmul(outputs[-1], weights) + biases

W_out = weight_variable([len(PARAMS)])
b_out = bias_variable([1])

y = RNN(x, W_out, b_out)
cost = tf.metrics.mean_squared_error(y_, y)
train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)

sess.run(tf.global_variables_initializer())

accuracyl = []
for b in range(100):
    for i in len(raw):
        train_step.run(feed_dict={x: raw[i, :-3], y_: raw[i, -2]})
    accuracy = tf.metrics.mean_squared_error(y_, y)
    accuracyl.append(accuracy.eval(feed_dict={x: raw[:, :-2], y_: raw[:, -1]}))
print accuracyl
