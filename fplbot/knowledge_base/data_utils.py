import requests
import pandas as pd
import numpy as np


def request_current_data():
    # This function requests current data from the FPL API
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    assert r.status_code == 200
    json = r.json()
    return json


def request_player_data(player_id):
    # This function requests data about a player from the FPL API
    url = (
        "https://fantasy.premierleague.com/api/element-summary/" + str(player_id) + "/"
    )
    r = requests.get(url)
    assert r.status_code == 200
    json = r.json()
    return json


def request_fixture_data():
    # This function requests fixture data from the FPL API
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url)
    assert r.status_code == 200
    json = r.json()
    return json


def process_current_data():
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
    elements_df_positions_teams["price"] = (
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

    # drop columns
    players_drop_cols = [
        "code",
        "cost_change_event",
        "cost_change_event_fall",
        "cost_change_start",
        "cost_change_start_fall",
        "element_type",
        "photo",
        "squad_number",
        "special",
        "team",
        "team_code",
        "transfers_in_event",
        "transfers_out_event",
        "web_name",
        "now_cost",
        "now_cost_rank",
        "now_cost_rank_type",
        "form_rank",
        "form_rank_type",
        "points_per_game_rank",
        "points_per_game_rank_type",
        "first_name",
        "second_name",
    ]
    elements_df_positions_teams = elements_df_positions_teams.drop(
        columns=players_drop_cols
    )

    # player_ids
    id2player = pd.Series(
        elements_df_positions_teams.full_name.values,
        index=elements_df_positions_teams.id,
    ).to_dict()

    # team ids
    id2team = pd.Series(teams_df_sub.name.values, index=teams_df_sub.id).to_dict()

    # map teams with names
    for col in ["most_selected", "most_captained", "most_vice_captained"]:
        events[col + "_team"] = events[col].apply(lambda x: id2team[x])

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

    data_dict = {}
    data_dict["players"] = elements_df_positions_teams
    data_dict["teams"] = teams_df_sub
    data_dict["id2player"] = id2player
    data_dict["gameweek_stats"] = events_sub
    data_dict["id2team"] = id2team

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
