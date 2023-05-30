import streamlit as st
from streamlit_chat import message
import sys

sys.path.append("../")

from fplbot.bot_agent.bot import fplAgent

st.title("Fantasy Premier League AssIstant Coach")

openai_api_key = st.text_input(
    "Enter your OpenAI API Key",
    key="user_key_input",
    type="password",
    autocomplete="current-password",
)

# initialize agent
agent = fplAgent(openai_api_key=openai_api_key)

# Storing the chat
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("Input question:", key="user_input")
    return input_text


# get user iput
user_input = get_text()

with st.container():
    # generate output and append to history
    if user_input:
        output = agent(user_input)
        # store the output
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
