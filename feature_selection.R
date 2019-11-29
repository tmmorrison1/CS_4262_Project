library(data.table)
library(ggplot2)

games = data.table(read.csv('league_data/LeagueofLegends.csv'))
all_teams = unlist(list(games[, blueTeamTag], games[, redTeamTag]))
num_teams = c()
num_games = c()



for(i in seq(0,100,5))
{
  factors = names(which(table(all_teams) > i))
  num_teams = c(num_teams, length(factors))
  total_games = sum(games[,redTeamTag]%in% factors & games[,blueTeamTag] %in% factors)
  num_games = c(num_games, total_games)
}

length(num_games) == length(num_teams)

## Make total games per minimum
p = qplot(seq(0,100,5), num_games) + labs(x = 'Minimum Games Played')
p = p + labs(y = 'Number of games in dataset')
p = p + labs(title = 'Total games as minimum required increased')
p = p + geom_line(color='red') 

## Make total teams per minimum
p = qplot(seq(0,100,5), num_teams) + labs(x = 'Minimum Games Played')
p = p + labs(y = 'Number of teams in dataset')
p = p + labs(title = 'Total teams as minimum required increased')
p = p + geom_line(color='red') 

mins = c()
maxs = c()
for(i in seq(0,100, 5))
{
  factors = names(which(table(all_teams) > i))
  tmp = games[redTeamTag %in% factors & blueTeamTag %in% factors]
  tmp_teams = unlist(list(tmp[, blueTeamTag], tmp[, redTeamTag]))
  team_table = table(tmp_teams)[which(table(tmp_teams) != 0)]
  mins = c(mins, min(team_table))
  maxs = c(maxs, max(team_table))
}


## Chopping minimum threshold of 45 games and 100 gamesplayed
factors = names(which(table(all_teams) > 45))
tmp = games[redTeamTag %in% factors & blueTeamTag %in% factors]
write.csv(tmp)





