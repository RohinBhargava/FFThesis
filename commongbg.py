from common import TEAM_DICT, YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np, pd, os, math

DICT_TEAM = {v : k for k, v in TEAM_DICT.items()}
DEF_PARAMS = ['PF','TYds','Ply','Y/P','TO','FL','1stD','Cmp','Att','Yds','TD','Int','NY/A','1stD','RAtt','RYds','RTD','Y/A','1stD','Pen','PYds','1stPy','Sc%','TO%']

positions = ['QB', 'RB', 'WR', 'TE']
running_average_position_year = []
season_totals_by_pos_year = []
lambda_m = 0.1

for pos in positions:
    season_totals_by_pos_year.append(np.float32(np.load('Data/serial/' + pos + str(YEAR_ST) + str(YEAR_END) + '.npy')))
    pos_list = []
    for year in range(YDIFF + 1):
        year_dict = dict()
        names = open('Data/Names/' + pos)
        for name in names:
            year_dict[name.strip()] = np.array([0.0] * (len(DEF_PARAMS) + len(PARAMS[pos])))
        pos_list.append(year_dict)
    running_average_position_year.append(pos_list)

defense_stats_by_year = []
for year in range(YEAR_ST, YEAR_END):
    defense = pd.read_csv('Data/Defense/' + str(year) + '.csv')
    def_dict = dict()
    teams = defense['Tm']
    stats = defense[DEF_PARAMS]/17
    for tm in range(len(teams)):
        def_dict[teams[tm]] = stats.ix[tm]
    defense_stats_by_year.append(def_dict)

for year in range(YEAR_ST, YEAR_END):
    year_han = year - YEAR_ST
    for week in range(1, 18):
        for game in os.listdir('Data/Game/' + str(year) + '/' + str(week)):
            print 'Data/Game/' + str(year) + '/' + str(week) + '/' + game
            teams = game.split('.csv')[0].split(' at ')
            abbr = map(lambda x : DICT_TEAM[x], teams)
            stats = pd.read_csv('Data/Game/' + str(year) + '/' + str(week) + '/' + game)
            names = [i.split('\\')[1] for i in stats['Player']]
            for name_i in range(len(names)):
                name = names[name_i]
                for ind in range(len(running_average_position_year)):
                    if name in running_average_position_year[ind][year_han]:
                        row = stats[PARAMS[positions[ind]]].ix[name_i]
                        for b in range(len(row)):
                            if (type(row[b]) != str) and math.isnan(row[b]):
                                row[b] = np.float32(0)
                            elif type(row[b]) == int:
                                row[b] = np.float32(row[b])
                        running_average_position_year[ind][year_han][name] *= lambda_m
                        running_average_position_year[ind][year_han][name] += (1 - lambda_m) * np.concatenate([np.array(row), -1 * np.array(defense_stats_by_year[year_han][teams[1 - abbr.index(stats['Tm'].ix[name_i])]])])
                        running_average_position_year[ind][year_han][name] /= int(week)
                        if week is '17' and year_han < YDIFF:
                            running_average_position_year[ind][year_han + 1][name] = running_average_position_year[ind][year_han][name]

for pos in running_average_position_year:
    print (pos[YDIFF])
