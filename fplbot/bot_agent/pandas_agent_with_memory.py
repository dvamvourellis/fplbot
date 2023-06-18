from typing import Any, Dict, List, Optional, Tuple
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.prompts.base import BasePromptTemplate
from langchain.tools.python.tool import PythonAstREPLTool
from fplbot.bot_agent.templates import MULTI_DF_PREFIX, SUFFIX_MULTI_DF
from langchain.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.llms.base import BaseLLM
from langchain.chains.llm import LLMChain
from langchain.callbacks.base import BaseCallbackManager
from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent


def get_multi_prompt_with_memory(
    dfs: List[Any],
    prefix: Optional[str] = MULTI_DF_PREFIX,
    suffix: Optional[str] = SUFFIX_MULTI_DF,
) -> Tuple[BasePromptTemplate, List[PythonAstREPLTool]]:
    num_dfs = len(dfs)

    input_variables = [
        "input",
        "chat_history",
        "agent_scratchpad",
        "num_dfs",
        "dfs_head",
    ]

    df_locals = {}
    for i, dataframe in enumerate(dfs):
        df_locals[f"df{i + 1}"] = dataframe
    tools = [PythonAstREPLTool(locals=df_locals)]

    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )

    partial_prompt = prompt.partial()
    if "dfs_head" in input_variables:
        dfs_head = "\n\n".join([d.head(1).to_markdown() for d in dfs])
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs), dfs_head=dfs_head)
    if "num_dfs" in input_variables:
        partial_prompt = partial_prompt.partial(num_dfs=str(num_dfs))
    return partial_prompt, tools


def create_pandas_multidf_with_memory(
    llm: BaseLLM,
    dfs: Any,
    prefix=MULTI_DF_PREFIX,
    suffix=SUFFIX_MULTI_DF,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = True,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 10,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""

    partial_prompt, tools = get_multi_prompt_with_memory(dfs, prefix, suffix)

    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )

    tool_names = [tool.name for tool in tools]

    agent = ZeroShotAgent(
        llm_chain=llm_chain,
        allowed_tools=tool_names,
        callback_manager=callback_manager,
        **kwargs,
    )
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        callback_manager=callback_manager,
        memory=memory
    )
