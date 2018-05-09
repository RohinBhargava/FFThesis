from common import TEAM_DICT, YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np, pd, os

DICT_TEAM = {v : k for k, v in TEAM_DICT.items()}
DEF_PARAMS = ['PF','TYds','Ply','Y/P','TO','FL','1stD','Cmp','Att','Yds','TD','Int','NY/A','1stD','RAtt','RYds','RTD','Y/A','1stD','Pen','PYds','1stPy','Sc%','TO%']

positions = ['QB', 'RB', 'WR', 'TE']
running_average_position_year = []
season_totals_by_pos_year = []

for pos in positions:
    season_totals_by_pos_year.append(np.float32(np.load('Data/serial/' + pos + str(YEAR_ST) + str(YEAR_END) + '.npy')))
    pos_list = []
    for year in range(YDIFF):
        year_dict = dict()
        names = open('Data/Names/' + pos)
        for name in names:
            year_dict[name] = [0] +
        pos_list.append(year_dict)
    running_average_position_year.append(pos_list)

defense_stats_by_year = []
for year in os.listdir('Data/Defense'):
    defense = pd.read_csv('Data/Defense/' + year)
    def_dict = dict()
    teams = defense['Tm']
    stats = defense[DEF_PARAMS]/17
    for tm in range(len(teams)):
        def_dict[teams[tm]] = stats[tm]
    defense_stats_by_year.append(def_dict)

for year in os.listdir('Data/Game'):
    year_han = int(year) - YEAR_ST
    for week in os.listdir('Data/Game/' + year):
        for game in os.listdir('Data/Game/' + year + '/' + week):
            teams = game.split('.csv')[0].split(' at ')
            abbr = map(lambda x : DICT_TEAM[x], teams)
            stats = pd.read_csv(game)
            names = stats['Player']
            for name in names:
                for ind in range(len(running_average_position_year)):
                    if name in i.split('\\')[1] in running_average_position_year[ind][year_han]:
                        if len(running_average_position_year[ind][year_han][i]) == 0:

                        else:
