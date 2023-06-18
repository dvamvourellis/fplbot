from pathlib import Path
from yaml import safe_load
import sys

sys.path.append("../")

config_path = Path("../config.yaml")
print(config_path)


def get_env_variable(key_name):
    with open(config_path, "r") as f:
        config = safe_load(f)
    env_value = config.get(key_name)
    if not env_value:
        raise ValueError("key not found in the configuration file")
    return env_value
