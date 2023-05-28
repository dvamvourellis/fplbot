MULTI_DF_PREFIX = """
You are working with {num_dfs} pandas dataframes in Python named df1, df2, df3, df4. \
Dataframe `df1` is useful for when you need to answer questions about each player using \
the most current view of his statistics and his current price. \
Dataframe `df2` is useful for when you need to answer questions about the past and 
future fixtures of each team and the difficulty of them. \
Dataframe `df3` is useful for when you need to answer questions about each team \
related to how strong each team is at home or away or in terms of attack and defence. \
Dataframe `df4` is is useful for when you need to answer questions about the statistics of \
each FPL gameweek such as the average score achieved and the player that was most captained \
or most selected.

You should use the tools below to answer the question posed of you:"""