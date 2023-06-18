import requests
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from fplbot.utils import get_env_variable
import os

DATA_DIR = get_env_variable("data_dir")


def request_current_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    json = r.json()
    return json


def request_player_data(player_id):
    url = (
        "https://fantasy.premierleague.com/api/element-summary/" + str(player_id) + "/"
    )
    r = requests.get(url)
    json = r.json()
    return json


def request_fixture_data():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url)
    json = r.json()
    return json


def process_current_data(update_data: bool = True):
    json = request_current_data()

    elements_df = pd.DataFrame(json["elements"])
    elements_types_df = pd.DataFrame(json["element_types"])
    teams_df = pd.DataFrame(json["teams"])
    events = pd.DataFrame(json["events"]).dropna()

    # merge position
    elements_df_positions = pd.merge(
        elements_df,
        elements_types_df[["id", "singular_name"]].rename(
            columns={"id": "element_type", "singular_name": "position"}
        ),
        how="inner",
        on=["element_type"],
    )

    # keep some team related stats to add for each player
    teams_df_sub = teams_df[
        [
            "id",
            "name",
            "strength",
            "strength_overall_home",
            "strength_overall_away",
            "strength_attack_home",
            "strength_attack_away",
            "strength_defence_home",
            "strength_defence_away",
        ]
    ]

    # merge team information for players
    elements_df_positions_teams = pd.merge(
        elements_df_positions,
        teams_df_sub[["id", "name"]].rename(
            columns={"id": "team", "name": "team_name"}
        ),
        how="inner",
        on=["team"],
    )

    # divide the price returned by 10 to show in millions
    elements_df_positions_teams["now_cost"] = (
        elements_df_positions_teams["now_cost"] / 10.0
    )

    # rename value to value_for_money
    elements_df_positions_teams["value_for_money"] = elements_df_positions_teams[
        "value_season"
    ].astype(float)
    elements_df_positions_teams = elements_df_positions_teams.drop(
        columns=["value_season"]
    )
    elements_df_positions_teams["full_name"] = (
        elements_df_positions_teams["first_name"]
        + " "
        + elements_df_positions_teams["second_name"]
    )
    elements_df_positions_teams = elements_df_positions_teams.rename(
        columns={"bonus": "bonus_points"}
    )

    # drop columns
    player_keep_cols = [
        "id",
        "position",
        "team_name",
        "second_name",
        "full_name",
        "now_cost",
        "form",
        "news",
        "points_per_game",
        "selected_by_percent",
        "value_for_money",
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus_points",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
    ]

    elements_df_positions_teams = elements_df_positions_teams[player_keep_cols]
    # convert non numeric columns
    for col in [
        "form",
        "selected_by_percent",
        "points_per_game",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
    ]:
        elements_df_positions_teams[col] = pd.to_numeric(
            elements_df_positions_teams[col], errors="coerce"
        )

    # player_ids
    id2player = pd.Series(
        elements_df_positions_teams.full_name.values,
        index=elements_df_positions_teams.id,
    ).to_dict()

    # team ids
    id2team = pd.Series(teams_df_sub.name.values, index=teams_df_sub.id).to_dict()

    # map events with names
    for col in ["most_selected", "most_captained", "most_vice_captained"]:
        events[col + "_name"] = events[col].apply(lambda x: id2player[x])

    events = events.rename(columns={"name": "gameweek_number"})
    events_sub = events[
        [
            "gameweek_number",
            "average_entry_score",
            "finished",
            "highest_score",
            "most_selected_name",
            "most_captained_name",
            "most_vice_captained_name",
        ]
    ]

    fixtures_df = get_fixture_data(id2team)

    data_dict = {}
    data_dict["players"] = elements_df_positions_teams
    data_dict["teams"] = teams_df_sub
    data_dict["id2player"] = id2player
    data_dict["gameweek_stats"] = events_sub
    data_dict["id2team"] = id2team
    data_dict["fixtures"] = fixtures_df

    if update_data == True:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(DATA_DIR + "players_dict.pickle", "wb") as f:
            pickle.dump(data_dict, f)

    return data_dict


def get_fixture_data(id2team):
    fixture_json = request_fixture_data()
    fixtures = pd.DataFrame(fixture_json)

    # map events with names
    for col in ["team_a", "team_h"]:
        fixtures[col + "_name"] = fixtures[col].apply(lambda x: id2team[x])

    fixtures_sub = fixtures[
        [
            "kickoff_time",
            "finished",
            "started",
            "team_a_score",
            "team_h_score",
            "team_h_difficulty",
            "team_a_difficulty",
            "team_a_name",
            "team_h_name",
        ]
    ]
    fixtures_sub = fixtures_sub.rename(
        columns={
            "team_a_score": "team_away_score",
            "team_h_score": "team_home_score",
            "team_h_difficulty": "team_home_difficulty",
            "team_a_difficulty": "team_away_difficulty",
            "team_a_name": "team_away_name",
            "team_h_name": "team_home_name",
        }
    )

    return fixtures_sub


def create_player_history(player_id, id2player):
    json = request_player_data(player_id)
    player_fixtures = pd.DataFrame(json["fixtures"])
    player_gw_history = pd.DataFrame(json["history"])
    player_history_past = pd.DataFrame(json["history_past"])

    player_data_dict = {}
    player_data_dict["player_fixtures"] = player_fixtures
    player_data_dict["player_gw_history"] = player_gw_history
    player_data_dict["player_history_past"] = player_history_past

    for df_key, df in player_data_dict.items():
        df["full_name"] = id2player[player_id]
        df["player_name"] = df["full_name"].str.split(" ").apply(lambda x: x[-1])

    return player_data_dict


def create_player_history_agg(id2player, id2team, update_data: bool = True):
    # create the data for all players
    player_data_dicts = Parallel(n_jobs=-1)(
        delayed(create_player_history)(player_id, id2player=id2player)
        for player_id in tqdm(id2player.keys())
    )

    player_history_past_agg = [
        player_data_dict["player_history_past"]
        for player_data_dict in player_data_dicts
    ]
    player_gw_history_agg = [
        player_data_dict["player_gw_history"] for player_data_dict in player_data_dicts
    ]
    player_history_past_agg = pd.concat(player_history_past_agg, axis=0)
    player_gw_history_agg = pd.concat(player_gw_history_agg, axis=0)

    # update price columns
    player_history_past_agg["start_cost"] = player_history_past_agg["start_cost"] / 10.0
    player_history_past_agg["end_cost"] = player_history_past_agg["end_cost"] / 10.0

    # update team name
    player_gw_history_agg["cost"] = player_gw_history_agg["value"] / 10.0
    player_gw_history_agg["opponent_team"] = player_gw_history_agg[
        "opponent_team"
    ].apply(lambda x: id2team[x])

    player_gw_history_agg_cols = [
        "full_name",
        "cost",
        "opponent_team",
        "total_points",
        "was_home",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ]
    player_gw_history_agg = player_gw_history_agg[player_gw_history_agg_cols]

    player_history_past_agg_cols = [
        "full_name",
        "season_name",
        "start_cost",
        "end_cost",
        "total_points",
        "minutes",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "own_goals",
        "penalties_saved",
        "penalties_missed",
        "yellow_cards",
        "red_cards",
        "saves",
        "bonus",
        "starts",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ]
    player_history_past_agg = player_history_past_agg[player_history_past_agg_cols]

    for col in [
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ]:
        player_gw_history_agg[col] = pd.to_numeric(
            player_gw_history_agg[col], errors="coerce"
        )
        player_history_past_agg[col] = pd.to_numeric(
            player_history_past_agg[col], errors="coerce"
        )

    player_data_agg_dict_final = {}
    player_data_agg_dict_final["players_season_history"] = player_history_past_agg
    player_data_agg_dict_final["players_gw_history"] = player_gw_history_agg

    if update_data == True:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(DATA_DIR + "players_history_dict.pickle", "wb") as f:
            pickle.dump(player_data_agg_dict_final, f)

    return player_data_agg_dict_final
