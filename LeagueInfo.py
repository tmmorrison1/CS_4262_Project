#!/usr/bin/env python3
import pandas as pd
import matplotlib as plt
import sys

name = '_min100'
def team_aggregate_diff(matches, save=False):
    global name
    matches['kill_diff_blue'] = matches.apply(lambda x: len(eval(x['bKills'])) - len(eval(x['rKills'])), axis = 1)
    matches['kill_diff_red'] = matches['kill_diff_blue'].apply(lambda x: -1 * x)
    matches['tower_diff_blue'] = matches.apply(lambda x: len(eval(x.bTowers)) - len(eval(x.rTowers)), axis = 1)
    matches['tower_diff_red'] = matches['tower_diff_blue'].apply(lambda x: -1 * x)
    matches['inhib_diff_blue'] = matches.apply(lambda x: len(eval(x.bInhibs)) - len(eval(x.rInhibs)), axis = 1)
    matches['inhib_diff_red'] = matches['inhib_diff_blue'].apply(lambda x: -1 * x)
    matches['dragon_diff_blue']= matches.apply(lambda x: len(eval(x.bDragons)) - len(eval(x.rDragons)), axis = 1)
    matches['dragon_diff_red'] = matches['dragon_diff_blue'].apply(lambda x: -1 * x)
    matches['baron_diff_blue'] = matches.apply(lambda x: len(eval(x.bBarons)) - len(eval(x.rBarons)), axis=1)
    matches['baron_diff_red'] = matches['baron_diff_blue'].apply(lambda x: -1 * x)
    matches['herald_diff_blue'] = matches.apply(lambda x: len(eval(x.bHeralds)) - len(eval(x.rHeralds)), axis=1)
    matches['herald_diff_red'] = matches['herald_diff_blue'].apply(lambda x: -1 * x)

    critical_cols_blue = matches[
        ['blueTeamTag', 'golddiff', 'kill_diff_blue', 'tower_diff_blue', 'inhib_diff_blue', 'dragon_diff_blue', 'baron_diff_blue', 'herald_diff_blue', 'bResult', 'Address']]
    critical_cols_red = matches[
        ['redTeamTag', 'golddiff', 'kill_diff_red', 'tower_diff_red', 'inhib_diff_red', 'dragon_diff_red', 'baron_diff_red', 'herald_diff_red', 'rResult', 'Address']]

    critical_cols_blue['summedGoldDiff'] = critical_cols_blue['golddiff'].apply(lambda x: sum(eval(x)))
    critical_cols_red['summedGoldDiff'] = critical_cols_red['golddiff'].apply(lambda x: sum(eval(x)))

    games_played_blue = critical_cols_blue.groupby('blueTeamTag').size().sort_index()
    games_played_red = critical_cols_red.groupby('redTeamTag').size().sort_index()
    blues_avg = critical_cols_blue.groupby('blueTeamTag').agg('sum').sort_index()
    blues_avg['count'] = games_played_blue
    red_avg = critical_cols_red.groupby('redTeamTag').agg('sum').sort_index()
    red_avg['count'] = games_played_red

    red_blue = red_avg.join(blues_avg, lsuffix='_red', rsuffix='_blue')
    red_blue['games'] = red_blue['count_red'] + red_blue['count_blue']
    red_blue['win_pct'] = (red_blue['rResult'] + red_blue['bResult']) / red_blue['games']
    red_blue['goldDiff'] = (-red_blue['summedGoldDiff_red'] + red_blue['summedGoldDiff_blue']) / red_blue['games']
    red_blue['kills'] = (red_blue['kill_diff_red'] + red_blue['kill_diff_blue']) / red_blue['games']
    red_blue['towers'] = (red_blue['tower_diff_red'] + red_blue['tower_diff_blue']) / red_blue['games']
    red_blue['inhibs'] = (red_blue['inhib_diff_red'] + red_blue['inhib_diff_blue']) / red_blue['games']
    red_blue['dragons'] = (red_blue['dragon_diff_red'] + red_blue['dragon_diff_blue']) / red_blue['games']
    red_blue['barons'] = (red_blue['baron_diff_red'] + red_blue['baron_diff_blue']) / red_blue['games']
    red_blue['heralds'] = (red_blue['herald_diff_red'] + red_blue['herald_diff_blue']) / red_blue['games']
    total_agg = red_blue[['games', 'win_pct', 'goldDiff', 'kills', 'towers', 'inhibs', 'dragons', 'barons', 'heralds']]

    if save:
        total_agg.rename_axis("team").to_csv('data/total_agg_diff' + name + '.csv')

    return total_agg


def team_aggregate(matches, save=False):
    global name

    critical_cols_blue = matches[['blueTeamTag', 'golddiff', 'bKills', 'bTowers', 'bInhibs', 'bDragons', 'bBarons', 'bHeralds', 'bResult', 'Address']]
    critical_cols_red = matches[['redTeamTag', 'golddiff',  'rKills', 'rTowers', 'rInhibs', 'rDragons','rBarons', 'rHeralds','rResult', 'Address']]

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
    critical_cols_blue['barons'] = critical_cols_blue['bBarons'].apply(lambda x: len(eval(x)))
    critical_cols_red['barons'] = critical_cols_red['rBarons'].apply(lambda x: len(eval(x)))
    critical_cols_blue['heralds'] = critical_cols_blue['bHeralds'].apply(lambda x: len(eval(x)))
    critical_cols_red['heralds'] = critical_cols_red['rHeralds'].apply(lambda x: len(eval(x)))

    games_played_blue =critical_cols_blue.groupby('blueTeamTag').size().sort_index()
    games_played_red = critical_cols_red.groupby('redTeamTag').size().sort_index()
    blues_avg = critical_cols_blue.groupby('blueTeamTag').agg('sum').sort_index()
    blues_avg['count'] = games_played_blue
    red_avg = critical_cols_red.groupby('redTeamTag').agg('sum').sort_index()
    red_avg['count'] = games_played_red

    red_blue = red_avg.join(blues_avg, lsuffix='_red', rsuffix='_blue')
    red_blue['games'] = red_blue['count_red'] + red_blue['count_blue']
    red_blue['win_pct'] = (red_blue['rResult'] + red_blue['bResult'])/red_blue['games']
    red_blue['goldDiff'] = (-red_blue['summedGoldDiff_red'] + red_blue['summedGoldDiff_blue'])/red_blue['games']
    red_blue['kills'] = (red_blue['kills_red'] + red_blue['kills_blue'])/red_blue['games']
    red_blue['towers'] = (red_blue['towers_red'] + red_blue['towers_blue'])/red_blue['games']
    red_blue['inhibs'] = (red_blue['inhibs_red'] + red_blue['inhibs_blue'])/red_blue['games']
    red_blue['dragons'] = (red_blue['dragons_red'] + red_blue['dragons_blue'])/red_blue['games']
    red_blue['barons'] = (red_blue['barons_red'] + red_blue['barons_blue']) / red_blue['games']
    red_blue['heralds'] = (red_blue['heralds_red'] + red_blue['heralds_blue']) / red_blue['games']
    total_agg = red_blue[['games', 'win_pct', 'goldDiff', 'kills', 'towers', 'inhibs', 'dragons', 'barons', 'heralds']]

    if save:
        total_agg.rename_axis("team").to_csv('data/total_agg'+name+'.csv')

    return total_agg

def make_plots(total_agg):
    global name
    path = 'figures/'
    kills = total_agg.plot.scatter(x='kills',
                                   y='win_pct',
                                   c='DarkBlue', title = 'Win Pct vs Kills')
    kills.set_xlabel('Kills Per Game')
    kills.set_ylabel('Win Percentage')
    kills.get_figure().savefig(path+'kill_corr'+name+'.png')
    

    goldDiff = total_agg.plot.scatter(x='goldDiff',
                                      y='win_pct',
                                      c='DarkBlue', title = 'Win Pct vs Gold Difference')
    goldDiff.set_xlabel('Gold Difference Per Game')
    goldDiff.set_ylabel('Win Percentage')
    goldDiff.get_figure().savefig(path+'goldDiff_corr'+name+'.png')
    
    towers = total_agg.plot.scatter(x='towers',
                                    y='win_pct',
                                    c='DarkBlue', title = 'Win Pct vs Towers Captured')
    towers.set_xlabel('Towers Captured Per Game')
    towers.set_ylabel('Win Percentage')
    towers.get_figure().savefig(path+'towers_corr'+name+'.png')

    dragons = total_agg.plot.scatter(x='dragons',
                                     y='win_pct',
                                     c='DarkBlue', title = 'Win Pct vs Dragons Slain' )
    dragons.set_xlabel('Dragons Slain Per Game')
    dragons.set_ylabel('Win Percentage')
    dragons.get_figure().savefig(path+'dragons_corr'+name+'.png')
    
    inhibs = total_agg.plot.scatter(x='inhibs',
                                    y='win_pct',
                                    c='DarkBlue', title = 'Win Pct vs Inhibitors Destroyed')
    inhibs.set_xlabel('Inhibitors Destroyed Per Game')
    inhibs.set_ylabel('Win Percentage')
    inhibs.get_figure().savefig(path+'inhibs_corr'+name+'.png')

    barons = total_agg.plot.scatter(x='barons',
                                    y='win_pct',
                                    c='DarkBlue', title='Win Pct vs Barons Killed')
    barons .set_xlabel('Barons Killed Per Game')
    barons .set_ylabel('Win Percentage')
    barons .get_figure().savefig(path + 'Barons_corr' + name + '.png')

    heralds = total_agg.plot.scatter(x='heralds',
                                    y='win_pct',
                                    c='DarkBlue', title='Win Pct vs Heralds Killed')
    heralds.set_xlabel('Heralds Killed Per Game')
    heralds.set_ylabel('Win Percentage')
    heralds.get_figure().savefig(path + 'Heralds_corr' + name + '.png')





if __name__ == '__main__':
    if len(sys.argv) > 3 or len(sys.argv) < 2:
        print('Usage: ./LeagueInfo.py <data_file_to_aggregate> [make_plots]')
        sys.exit(1)
    path = sys.argv[1]
    plots = False if len(sys.argv) != 3 else True

    aggregated = team_aggregate(path)

    if plots:
        make_plots(aggregated)
