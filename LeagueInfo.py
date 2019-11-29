import pandas as pd
import matplotlib as plt


#This is embarrasingly bad practice :(
matches = pd.read_csv('~/Documents/CS_4262_Project/league_data/LeagueofLegends.csv')

critical_cols_blue = matches[['blueTeamTag', 'golddiff', 'bKills', 'bTowers', 'bInhibs', 'bDragons', 'bResult', 'Address']]
critical_cols_red = matches[['redTeamTag', 'golddiff',  'rKills', 'rTowers', 'rInhibs', 'rDragons','rResult', 'Address']]

critical_cols_blue['summedGoldDiff'] = critical_cols_blue['golddiff'].apply(lambda x: sum(eval(x)))
critical_cols_red['summedGoldDiff'] = critical_cols_red['golddiff'].apply(lambda x: sum(eval(x)))
critical_cols_blue['kills'] = critical_cols_blue['bKills'].apply(lambda x: len(eval(x)))
critical_cols_red['kills'] = critical_cols_red['rKills'].apply(lambda x: len(eval(x)))
critical_cols_blue['towers'] = critical_cols_blue['bTowers'].apply(lambda x: len(eval(x)))
critical_cols_red['towers'] = critical_cols_red['rTowers'].apply(lambda x: len(eval(x)))
critical_cols_blue['inhibs'] = critical_cols_blue['bInhibs'].apply(lambda x: len(eval(x)))
critical_cols_red['inhibs'] = critical_cols_red['rInhibs'].apply(lambda x: len(eval(x)))
critical_cols_blue['dragons'] = critical_cols_blue['bDragons'].apply(lambda x: len(eval(x)))
critical_cols_red['dragons'] = critical_cols_red['rDragons'].apply(lambda x: len(eval(x)))

games_played_blue =critical_cols_blue.groupby('blueTeamTag').size().sort_index()
games_played_red = critical_cols_red.groupby('redTeamTag').size().sort_index()
blues_avg = critical_cols_blue.groupby('blueTeamTag').agg('sum').sort_index()
blues_avg['count'] = games_played_blue
blues_avg.to_csv('blue_avg.csv')
red_avg = critical_cols_red.groupby('redTeamTag').agg('sum').sort_index()
red_avg['count'] = games_played_red
red_avg.to_csv('red_avg.csv')

red_blue = red_avg.join(blues_avg, lsuffix='_red', rsuffix='_blue')
red_blue['games'] = red_blue['count_red'] + red_blue['count_blue']
red_blue['win_pct'] = (red_blue['rResult'] + red_blue['bResult'])/red_blue['games']
red_blue['goldDiff'] = (-red_blue['summedGoldDiff_red'] + red_blue['summedGoldDiff_blue'])/red_blue['games']
red_blue['kills'] = (red_blue['kills_red'] + red_blue['kills_blue'])/red_blue['games']
red_blue['towers'] = (red_blue['towers_red'] + red_blue['towers_blue'])/red_blue['games']
red_blue['inhibs'] = (red_blue['inhibs_red'] + red_blue['inhibs_blue'])/red_blue['games']
red_blue['dragons'] = (red_blue['dragons_red'] + red_blue['dragons_blue'])/red_blue['games']
total_agg = red_blue[['games', 'win_pct', 'goldDiff', 'kills', 'towers', 'inhibs', 'dragons']]
total_agg.rename_axis("team").to_csv('total_agg.csv')


kills = total_agg.plot.scatter(x='kills',
                     y='win_pct',
                    c='DarkBlue', title = 'Win Pct vs Kills')
kills.set_xlabel('Kills Per Game')
kills.set_ylabel('Win Percentage')
kills.get_figure().savefig('kill_corr.png')


goldDiff = total_agg.plot.scatter(x='goldDiff',
                     y='win_pct',
                    c='DarkBlue', title = 'Win Pct vs Gold Difference')
goldDiff.set_xlabel('Gold Difference Per Game')
goldDiff.set_ylabel('Win Percentage')
goldDiff.get_figure().savefig('goldDiff_corr.png')

towers = total_agg.plot.scatter(x='towers',
                     y='win_pct',
                    c='DarkBlue', title = 'Win Pct vs Towers Captured')
towers.set_xlabel('Towers Captured Per Game')
towers.set_ylabel('Win Percentage')
towers.get_figure().savefig('towers_corr.png')

dragons = total_agg.plot.scatter(x='dragons',
                     y='win_pct',
                    c='DarkBlue', title = 'Win Pct vs Dragons Slain' )
dragons.set_xlabel('Dragons Slain Per Game')
dragons.set_ylabel('Win Percentage')
dragons.get_figure().savefig('dragons_corr.png')

inhibs = total_agg.plot.scatter(x='inhibs',
                     y='win_pct',
                    c='DarkBlue', title = 'Win Pct vs Inhibitors Destroyed')
inhibs.set_xlabel('Inhibitors Destroyed Per Game')
inhibs.set_ylabel('Win Percentage')
inhibs.get_figure().savefig('inhibs_corr.png')
