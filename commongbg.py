from common import TEAM_DICT, YEAR_ST, YEAR_END, PARAMS, YDIFF, allDataParse, np, pd, os

DICT_TEAM = {v : k for k, v in TEAM_DICT.items()}

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
            year_dict[name] = []
        pos_list.append(year_dict)
    season_totals_by_pos_year.append(pos_list)

for year in os.listdir('Data/Game'):
    for week in os.listdir('Data/Game/' + year):
        for game in 
