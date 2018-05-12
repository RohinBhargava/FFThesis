from common import TEAM_DICT, PARAMS, YDIFF, np, pd, os, math
import pdb

DICT_TEAM = {v : k for k, v in TEAM_DICT.items()}
DEF_PARAMS = ['PF','TYds','Ply','Y/P','TO','FL','1stD','Cmp','Att','Yds','TD','Int','NY/A','1stD','RAtt','RYds','RTD','Y/A','1stD','Pen','PYds','1stPy','Sc%','TO%']

positions = ['QB', 'RB', 'WR', 'TE']

def generate(start, end, names_l, lambda_m):
    running_average_position_year = []
    actuals = []
    for pos in positions:
        pos_list = []
        actuals_list = []
        for year in range(end - start):
            year_dict = dict()
            actuals_dict = dict()
            for name in names_l[pos]:
                # year_dict[name.strip()] = np.array([0.0] * (len(DEF_PARAMS) + len(PARAMS[pos])))
                year_dict[name.strip()] = []
                actuals_dict[name.strip()] = []
            pos_list.append(year_dict)
            actuals_list.append(actuals_dict)
        running_average_position_year.append(pos_list)
        actuals.append(actuals_list)

    defense_stats_by_year = []
    for year in range(start, end):
        defense = pd.read_csv('Data/Defense/' + str(year) + '.csv')
        def_dict = dict()
        teams = defense['Tm']
        stats = defense[DEF_PARAMS]/17
        for tm in range(len(teams)):
            def_dict[teams[tm]] = stats.ix[tm]
        defense_stats_by_year.append(def_dict)

    for year in range(start, end):
        year_han = year - start
        for week in range(1, 18):
            for game in os.listdir('Data/Game/' + str(year) + '/' + str(week)):
                print ('Data/Game/' + str(year) + '/' + str(week) + '/' + game)
                teams = game.split('.csv')[0].split(' at ')
                abbr = list(map(lambda x : DICT_TEAM[x], teams))
                stats = pd.read_csv('Data/Game/' + str(year) + '/' + str(week) + '/' + game)
                names = [i.split('\\')[1] for i in stats['Player']]
                for name_i in range(len(names)):
                    name = names[name_i]
                    for ind in range(len(running_average_position_year)):
                        if name in running_average_position_year[ind][year_han]:
                            pos = positions[ind]
                            row = stats[PARAMS[positions[ind]]].ix[name_i]
                            for b in range(len(row)):
                                if (type(row[b]) != str) and math.isnan(row[b]):
                                    row[b] = np.float32(0)
                                elif type(row[b]) == int:
                                    row[b] = np.float32(row[b])
                            concatrow = np.concatenate([np.array(row), -1 * np.array(defense_stats_by_year[year_han][teams[1 - abbr.index(stats['Tm'].ix[name_i])]])])
                            if week == 1:
                                st = np.array([0.0] * (len(DEF_PARAMS) + len(PARAMS[pos])))
                                if year != start:
                                    st = running_average_position_year[ind][year_han - 1][name][-1]
                                running_average_position_year[ind][year_han][name].append(st)
                            old = running_average_position_year[ind][year_han][name][-1]
                            new = (lambda_m * old * len(old) + (1 - lambda_m) * concatrow)/(1 + len(old))
                            if week != 17:
                                running_average_position_year[ind][year_han][name].append(new)
                            actuals[ind][year_han][name].append(row)

            for ind in range(len(running_average_position_year)):
                for name in running_average_position_year[ind][year_han]:
                    if len(running_average_position_year[ind][year_han][name]) < week:
                        pos = positions[ind]
                        row = np.array([0.0] * len(PARAMS[pos]))
                        concatrow = np.array([0.0] * (len(DEF_PARAMS) + len(PARAMS[pos])))
                        if week == 1:
                            st = np.array([0.0] * (len(DEF_PARAMS) + len(PARAMS[pos])))
                            if year != start:
                                st = running_average_position_year[ind][year_han - 1][name][-1]
                            running_average_position_year[ind][year_han][name].append(st)
                        old = running_average_position_year[ind][year_han][name][-1]
                        new = (lambda_m * old * len(old) + (1 - lambda_m) * concatrow)/(1 + len(old))
                        if week != 17:
                            running_average_position_year[ind][year_han][name].append(new)
                        actuals[ind][year_han][name].append(row)

    return running_average_position_year, actuals

def allDataParse(start, end, pos, lambda_m=0.1):
    names = dict()
    for pos_i in positions:
        names[pos_i] = open('Data/Names/' + pos_i, 'r').read().splitlines()
    if not os.path.exists('Data/serial/Games/' + pos + str(start) + str(end) + str(lambda_m) + '.npy'):
        r, t = generate(start, end, names, lambda_m)
        array_han = []
        actual_han = []
        for pos_i in positions:
            pos_arr = np.zeros((len(names[pos_i]), end - start, 17, len(DEF_PARAMS) + len(PARAMS[pos])))
            actual_arr = np.zeros((len(names[pos_i]), end - start, 17, len(PARAMS[pos])))
            pos_name = names[pos_i]
            for na in range(len(pos_name)):
                pl_na = pos_name[na]
                pos_y = r[positions.index(pos_i)]
                pos_y_acc = t[positions.index(pos_i)]
                for year in range(len(pos_y)):
                    week = pos_y[year][pl_na]
                    week_acc = pos_y_acc[year][pl_na]
                    for wk in range(len(week)):
                        stats = week[wk]
                        stats_acc = week_acc[wk]
                        for stat in range(len(stats)):
                            pos_arr[na, year, wk, stat] = stats[stat]
                        for stat in range(len(stats_acc)):
                            actual_arr[na, year, wk, stat] = stats_acc[stat]
            array_han.append(pos_arr)
            actual_han.append(actual_arr)
        for pos_i in range(len(array_han)):
            np.save('Data/serial/Games/' + positions[pos_i] + str(start) + str(end) + str(lambda_m) + '.npy', array_han[pos_i])
            np.save('Data/serial/Games/Actuals/' + positions[pos_i] + str(start) + str(end) + '.npy', actual_han[pos_i])
    data = np.load('Data/serial/Games/' + pos + str(start) + str(end) + str(lambda_m) + '.npy')
    actuals = np.load('Data/serial/Games/Actuals/' + pos + str(start) + str(end) + '.npy')
    mean = np.mean(data, axis = 0)
    mean_acc = np.mean(actuals, axis = 0)
    std = np.std(data, axis = 0)
    std_acc = np.std(actuals, axis = 0)
    for i in range(len(data)):
        data[i] -= mean
        actuals[i] -= mean_acc
        for j in range(len(std)):
            for k in range(len(std[j])):
                for l in range(len(std[j, k])):
                    if std[j, k, l] > 1:
                        data[i, j, k, l] /= std[j, k, l]
        for j in range(len(std_acc)):
            for k in range(len(std_acc[j])):
                for l in range(len(std_acc[j, k])):
                    if std_acc[j, k, l] > 1:
                        actuals[i, j, k, l] /= std_acc[j, k, l]
    return data, mean, std, actuals, mean_acc, std_acc, names[pos]
