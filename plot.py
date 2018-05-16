from matplotlib import pyplot as plt
from common import YEAR_ST, YEAR_END, TOP_FIVE, PARAMS
import common, commongbg, pandas as pd

positions = ['QB', 'RB', 'WR', 'TE']
stats_to_acc = {'QB' : 3, 'RB' : 2, 'WR' : 6, 'TE' : 3}
stat_names =  {'QB' : 'Passing Touchdowns', 'RB' : 'Rushing Touchdowns', 'WR' : 'Receiving Touchdowns', 'TE' : 'Receiving Touchdowns'}

f, axarr = plt.subplots(len(positions), 5)
r = 0
seasons_plts = []
games_plts = []
for pos in positions:
    seasons, smean, sstd, names = common.allDataParse(YEAR_ST,YEAR_END, pos)
    raw, mean, std, acs, mean_acc, std_acc, _ = commongbg.allDataParse(YEAR_ST,YEAR_END, pos)
    nopl, years, weeks, stats = raw.shape
    d_slice = [names.index(i) for i in TOP_FIVE[pos]]
    for i in range(5):
        season_line = [0]
        game_line = [0]
        for j in range(years):
            y = pd.read_csv('Data/' + str(j + 2010) + '.csv')
            ycodes = [i.split('\\')[1] for i in y['Name']]
            season_div = 0.0
            name = TOP_FIVE[pos][i]
            if name in ycodes:
                season_div = y[PARAMS[pos][stats_to_acc[pos]]][ycodes.index(name)]/weeks
            for k in range(weeks):
                season_line.append(season_line[-1] + season_div)
                game_line.append(game_line[-1] + acs[d_slice[i], j, k, stats_to_acc[pos]] * std_acc[j, k, stats_to_acc[pos]] + mean[j, k, stats_to_acc[pos]])
        axarr[r][i].plot([m for m in range(weeks * years + 1)], season_line, color='blue')
        axarr[r][i].plot([m for m in range(weeks * years + 1)], game_line, color='red')
        axarr[r][i].set_title(TOP_FIVE[pos][i] + ' Career ' + stat_names[pos] + ' by Week since 2010')
        axarr[r][i].set_ylabel(stat_names[pos])
        # axarr[i][r].set_xlabel('Week since the 2010 Season Started')
    r += 1

plt.show()
