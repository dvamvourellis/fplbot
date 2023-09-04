MULTI_DF_PREFIX_WITH_HIST = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, df3, df4, df5, df6.
Dataframe `df1` is useful for when you need to answer questions about each player using \
the most current view of his aggregate statistics up to this gameweek and his current cost. 
Dataframe `df2` is useful for when you need to answer questions about the past and \
future fixtures of each team and the difficulty of them. 
Dataframe `df3` is useful for when you need to answer questions \
related to how strong each team is at home or away or in terms of attack and defence.
Dataframe `df4` is useful for when you need to answer questions about the statistics of \
each FPL gameweek such as the average score achieved by users and the player that was most captained \
or most selected.
Dataframe `df5` is useful for when you need to answer questions about the total statistics of \
any player for previous seasons. 
Dataframe `df6` is useful for when you need to answer questions about the statistics of \
any player for previous gameweeks from this season. The columns of this dataframe reflect the statistics \
achieved by each player in each individual gameweek.
You should use the tools below to answer the question posed of you:
    """

MULTI_DF_PREFIX = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, df3, df4.
Dataframe `df1` is useful for when you need to answer questions about each player using \
the most current view of his aggregate statistics up to this gameweek and his current cost. 
Dataframe `df2` is useful for when you need to answer questions about the past and \
future fixtures of each team and the difficulty of them. 
Dataframe `df3` is useful for when you need to answer questions \
related to how strong each team is at home or away or in terms of attack and defence.
Dataframe `df4` is useful for when you need to answer questions about the statistics of \
each FPL gameweek such as the average score achieved by users and the player that was most captained \
or most selected.
You should use the tools below to answer the question posed of you:
    """

SUFFIX_MULTI_DF = """
This is the result of `print(df.head(1))` for each dataframe:
{dfs_head}

Begin!
{chat_history}
Question: {input}
{agent_scratchpad}"""
