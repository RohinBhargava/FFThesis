import pandas as pd, numpy as np, math, random, sys

TEAM_DICT = {'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens', 'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears', 'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys', 'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GNB': 'Green Bay Packers', 'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars', 'KAN': 'Kansas City Chiefs', 'LAC': 'Los Angeles Chargers', 'LAR': 'Los Angeles Rams', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings', 'NOR': 'New Orleans Saints', 'NWE': 'New England Patriots', 'NYG': 'New York Giants', 'NYJ': 'New York Jets', 'OAK': 'Oakland Raiders', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers', 'SDG': 'San Diego Chargers', 'SEA': 'Seattle Seahawks', 'SFO': 'San Francisco 49ers', 'STL': 'St. Louis Rams', 'TAM': 'Tampa Bay Buccaneers', 'TEN': 'Tennessee Titans', 'WAS': 'Washington Redskins'}
PARAMS = ['Tm', 'FantPos', 'Age', 'G', 'GS', 'Cmp', 'Att', 'Yds', 'TD', 'Int', 'RAtt', 'RYds', 'Y/A', 'RTD', 'Tgt', 'Rec', 'WYds', 'Y/R', 'WTD', 'FantPt', 'DKPt', 'FDPt', 'PosRank']
YEAR_ST = 2006
YEAR_END = 2018
YDIFF = YEAR_END - YEAR_ST - 1

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
    table = y[PARAMS]
    ranks = rank(year)
    for i in range(len(ycodes)):
        if table['FantPos'][i] == pos:
            player = ycodes[i]
            if player not in tsd:
                tsd[player] = []
                while len(tsd[player]) < n - 1:
                    tsd[player].append([0] * len(PARAMS))
            row = table.ix[i]
            for b in range(len(row)):
                if (type(row[b]) != str) and math.isnan(row[b]):
                    row[b] = 0.0
                elif type(row[b]) == int:
                    row[b] = float(row[b])
            row[1] = 1.0
            if 'TM' not in table['Tm'][i]:
                row[0] = 32 - ranks.index(TEAM_DICT[table['Tm'][i]])
            else:
                row[0] = 32/int(table['Tm'][i][0])
            tsd[player].append(row)
    for i in tsd:
        if len(tsd[i]) < n:
            tsd[i].append([0] * len(PARAMS))
    return tsd

def allDataParse(start, end):
    tsd = dict()
    for i in range(start, end):
        tsd = parsePl(i, tsd, i - start + 1, sys.argv[1])
    darr = []
    for i in tsd:
        darr.append(tsd[i])
    return np.array(darr)