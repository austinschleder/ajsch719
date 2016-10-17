from __future__ import print_function
import nfldb
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from pandas.stats.api import ols
import scipy
import sys

db = nfldb.connect()

def update_completed_games(season):
	q = nfldb.Query(db)
	q.game(season_year=season, season_type='Regular')
	all_games = q.sort(('gsis_id', 'asc')).as_games()
	game_data = [[g.week, g.home_team, g.away_team, g.home_score - g.away_score] for g in all_games]
	scores = pd.DataFrame(game_data, columns=['week', 'home_team', 'away_team', 'point_diff'])
	scores.to_csv('app/static/{}_all_games.csv'.format(season), index=False)

def get_all_teams(season):
	games = pd.read_csv('app/static/{}_all_games.csv'.format(season), header=0)
	all_teams = sorted(list(set(list(games['home_team']) + list(games['away_team']))))
	return all_teams

def create_formula_matrix(season, target_week):
	all_teams = get_all_teams(season)
	all_games = pd.read_csv('app/static/{}_all_games.csv'.format(season), header=0)
	completed_games = all_games[all_games['week'] < target_week]
	point_diffs = completed_games['point_diff']
	formula_matrix = pd.DataFrame(columns=all_teams)
	for team in all_teams:
		formula_matrix[team] = np.where(completed_games['home_team'] == team, 1, np.where(completed_games['away_team'] == team, -1, 0))
	return formula_matrix, point_diffs

def calculate_rankings(season, target_week):
	formula_matrix, point_diffs = create_formula_matrix(season, target_week)
	res = ols(y=point_diffs, x=formula_matrix.drop(['ARI'], axis=1))
	q_scores = res.beta[0:-1]
	q_scores['ARI'] = 0
	q_scores -= min(res.beta)
	q_scores.sort_values(ascending=False, inplace=True)
	q_scores = pd.DataFrame({'team':q_scores.index, 'flower_power':q_scores.values})
	q_scores.to_csv('app/static/{}_{}_flowerpower.csv'.format(str(season), str(target_week).zfill(2)), float_format='%.2f', index=False)
	hfa = pd.Series(res.beta['intercept'])
	hfa.to_csv('app/static/{}_{}_hfa.csv'.format(str(season), str(target_week).zfill(2)), float_format='%.2f')

if __name__ == '__main__':
	seasons = [2016]
	weeks = [5, 6]
	for season in seasons:
		update_completed_games(season)
		for target_week in weeks:
			calculate_rankings(season, target_week)