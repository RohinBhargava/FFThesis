import tensorflow as tf, pandas as pd, numpy as np, math, random, sys, sklearn.linear_model, os

team_dict = {'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GNB': 'Green Bay Packers', 'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KAN': 'Kansas City Chiefs', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings', 'NOR': 'New Orleans Saints', 'NWE': 'New England Patriots', 'NYG': 'New York Giants', 'NYJ': 'New York Jets', 'OAK': 'Oakland Raiders', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SDG': 'San Diego Chargers', 'SEA': 'Seattle Seahawks', 'SFO': 'San Francisco 49ers', 'STL': 'St. Louis Rams', 'TAM': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Redskins'}
params = ['Tm', 'FantPos', 'Age', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'Att', 'Yds', 'Y/A', 'TD', 'Tgt', 'Rec', 'Yds', 'Y/R', 'TD', 'FantPt', 'DKPt', 'FDPt', 'PosRank']
totals = [0] * len(params)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def rank(year):
    nfc = pd.read_csv('Data/NFC' + str(year) + '.csv')
    afc = pd.read_csv('Data/AFC' + str(year) + '.csv')
    ranks = []
    i = 0
    j = 0
    while i < len(nfc['Tm']) or j < len(afc['Tm']):
        if i == len(nfc['Tm']):
            ranks.append(afc['Tm'][j])
            j += 1
        elif j == len(afc['Tm']):
            ranks.append(nfc['Tm'][i])
            i += 1
        elif afc['W-L%'][j] > nfc['W-L%'][i]:
            ranks.append(afc['Tm'][j])
            j += 1
        elif afc['W-L%'][j] == nfc['W-L%'][i]:
            if afc['SRS'][j] > nfc['SRS'][i]:
                ranks.append(afc['Tm'][j])
                j += 1
            else:
                ranks.append(nfc['Tm'][i])
                i += 1
        else:
            ranks.append(nfc['Tm'][i])
            i += 1
    return ranks

def parsePl(year, tsd, n, pos):
    y = pd.read_csv('Data/' + str(year) + '.csv')
    ycodes = [i.split('\\')[1] for i in y['Name']]
    table = y[params]
    for i in range(len(ycodes)):
        if table['FantPos'][i] == pos:
            player = ycodes[i]
            if player not in tsd:
                tsd[player] = []
            row = table.ix[i]
            for b in range(len(row)):
                if type(row[b]) == float and math.isnan(row[b]):
                    row[b] = 0.0
                elif type(row[b]) == int:
                    row[b] = float(row[b])
            row[1] = 1.0
            if 'TM' not in table['Tm'][i]:
                row[0] = 32 - rank(year).index(team_dict[table['Tm'][i]])
            else:
                row[0] = 32/int(table['Tm'][i][0])
            tsd[player].append(row)
            for b in range(len(row)):
                totals[b] += row[b]
    for i in tsd:
        while len(tsd[i]) < n:
            tsd[i].append([0] * len(params))
    return tsd

def allDataParse(start, end):
    tsd = dict()
    for i in range(start, end):
        tsd = parsePl(i, tsd, i - start + 1, sys.argv[1])
    darr = []
    for i in tsd:
        darr.append(tsd[i])
    return np.array(darr)


sess = tf.InteractiveSession()
raw = allDataParse(2012,2018)

lr = sklearn.linear_model.LinearRegression()
b = [0] * len(params)
for i in range(5):
    a = lr.fit(raw[:, i], raw[:, i + 1])
    b += np.mean(np.square(a.predict(raw[:, i]) - raw[:, i + 1]), axis=0)
print zip(params,b/5)
print np.mean(b)/5

# x = tf.placeholder(tf.float32, shape=[len(raw), len(params)])
# y_ = tf.placeholder(tf.float32, shape=[len(raw), len(params)])
#
# def weight_variable(shape):
#   initial = tf.truncated_normal(shape, stddev=0.1)
#   return tf.Variable(initial)
#
# def bias_variable(shape):
#   initial = tf.constant(0.1, shape=shape)
#   return tf.Variable(initial)
#
# # THIS IS TECHNICALLY TWO LAYERS, MAKE 1
# # TIME SERIES STRATEGIES
#
# W_input = weight_variable([len(params), len(params)])
# b_input = bias_variable([len(params)])
# # W = tf.Variable(np.random.randn(), name="weight")
# # b = tf.Variable(np.random.randn(), name="bias")
#
# # h1 = tf.matmul(x,W_input) + b_input
# y = tf.matmul(x,W_input) + b_input
#
# # W_out = weight_variable([len(raw), len(params)])
# # b_out = bias_variable([len(params)])
# #
# # y = tf.matmul(h1,W_out) + b_out
# cost = tf.reduce_mean(tf.square(y - y_))
# train_step = tf.train.AdamOptimizer(0.0001).minimize(cost)
#
# sess.run(tf.global_variables_initializer())
#
# accuracyl = []
#
# for e in range(10):
#     for i in range(4):
#         train_step.run(feed_dict={x: raw[:, i], y_: raw[:, i + 1]})
#     # MEAN SQUARED ERROR AS EVALUATION METRIC
#     accuracy = tf.reduce_mean(tf.square(y - y_))
#     print e, ':', accuracy.eval(feed_dict={x: raw[:, -2], y_: raw[:, -1]})

# SCIKIT SKLEARN LINEAR REGRESSION, LOOK AT THIS: L1 AND L2 REGRESSION (L2 WILL PROBABLY GIVE BETTER RESULTS)

