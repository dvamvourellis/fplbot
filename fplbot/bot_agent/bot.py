import os
from typing import Optional, Union, Dict, Any
import sys

sys.path.append("../")

from fplbot.bot_agent.llm_constructors import init_davinci, init_gpt35_turbo
from fplbot.knowledge_base.data_utils import process_current_data, get_fixture_data
from fplbot.bot_agent.templates import MULTI_DF_PREFIX

from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import create_pandas_dataframe_agent


class fplAgent:
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: Optional[str] = "text-davinci-003",
        return_intermediate_steps: bool = True,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps

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
        data_dict = process_current_data()
        self.teams_df = data_dict["teams"]
        self.players_df = data_dict["players"]
        self.gameweek_stats_df = data_dict["gameweek_stats"]
        self.fixtures_df = get_fixture_data(data_dict["id2team"])

    def _init_agent(self):
        dfs = [self.players_df, self.fixtures_df, self.teams_df, self.gameweek_stats_df]
        self.agent = create_pandas_dataframe_agent(
            self.llm,
            dfs,
            verbose=True,
            prefix=MULTI_DF_PREFIX,
            include_df_in_prompt=self.verbose,
            return_intermediate_steps=self.return_intermediate_steps,
            max_iterations=5,
            max_execuion_time=None,
        )
