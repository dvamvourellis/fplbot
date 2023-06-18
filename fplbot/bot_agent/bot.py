import os
from typing import Optional, Union, Dict, Any
import sys
import pickle

sys.path.append("../")

from fplbot.bot_agent.llm_constructors import init_davinci, init_gpt35_turbo
from fplbot.knowledge_base.data_utils import process_current_data, get_fixture_data, create_player_history_agg
from fplbot.utils import get_env_variable
from fplbot.bot_agent.pandas_agent_with_memory import create_pandas_multidf_with_memory

from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import create_pandas_dataframe_agent

DATA_DIR = get_env_variable('data_dir')

class fplAgent:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: Optional[str] = "gpt-3.5-turbo",
        update_data=False,
        return_intermediate_steps: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps
        self.update_data = update_data

        self._init_llm(openai_api_key, model_name)

        self._fetch_data()

        self._init_agent()

    def __call__(self, text: str) -> str:
        response = self.agent(text)["output"]
        return response

    def _init_llm(
        self,
        openai_api_key: Optional[str] = None,
        model_name: Optional[str] = "gpt-3.5-turbo",
    ):
        if model_name == "gpt-3.5-turbo":
            self.llm = init_gpt35_turbo(openai_api_key=openai_api_key)
        elif model_name == "text-davinci-003":
            self.llm = init_davinci(openai_api_key=openai_api_key)
        else:
            raise ValueError(
                f"Please provide valid OpenAI model name, got {model_name}"
            )

    def _fetch_data(self):
        if self.update_data==True:
            data_dict = process_current_data(update_data=self.update_data)
            player_data_agg_dict_final = create_player_history_agg(data_dict['id2player'], data_dict['id2team'], update_data=self.update_data)
        else:
            with open(DATA_DIR+'players_dict.pickle', "rb") as f:
                data_dict = pickle.load(f)
            with open(DATA_DIR+'players_history_dict.pickle', "rb") as f:
                player_data_agg_dict_final = pickle.load(f)

        self.teams_df = data_dict["teams"]
        self.players_df = data_dict["players"]
        self.gameweek_stats_df = data_dict["gameweek_stats"]
        self.fixtures_df = data_dict["fixtures"]

        self.players_season_history = player_data_agg_dict_final['players_season_history']
        self.players_gw_history = player_data_agg_dict_final['players_gw_history']

    def _init_agent(self):
        dfs = [self.players_df, self.fixtures_df, self.teams_df, self.gameweek_stats_df, 
               self.players_season_history, self.players_gw_history]
        
        self.agent = create_pandas_multidf_with_memory(
            self.llm, 
            dfs, 
            verbose=self.verbose
            )