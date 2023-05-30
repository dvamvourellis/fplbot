import os
import sys
from typing import Optional, Union, Dict, Any


from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

sys.path.append("../")
from fplbot.utils import get_env_variable

# os.environ['OPENAI_API_KEY'] = get_env_variable('openai_api_key')


def init_davinci(openai_api_key: Optional[str] = None, temperature=0):
    if openai_api_key is None and "OPENAI_API_KEY" not in os.environ:
        raise Exception("No OpenAI API key provided")
    openai_api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
    # instantiate the OpenAI API wrapper
    llm = OpenAI(
        model_name="text-davinci-003", openai_api_key=openai_api_key, temperature=0.0
    )
    return llm


def init_gpt35_turbo(openai_api_key: Optional[str] = None, temperature=0):
    if openai_api_key is None and "OPENAI_API_KEY" not in os.environ:
        raise Exception("No OpenAI API key provided")
    openai_api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
    # instantiate the OpenAI API wrapper
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.0
    )
    return llm


def init_gpt4all(model_binary_name):
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(
        model=os.environ["MODEL_PATH"] + model_binary_name,
        n_ctx=1000,
        backend="gptj",
        callbacks=callbacks,
        verbose=False,
    )
    return llm
