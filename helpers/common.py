from configparser import ConfigParser
import os

def get_env_var(group, var_name): 
    config = ConfigParser()
    file_path = ".env"
    if os.path.exists(file_path):
        config.read(file_path)
        return  config[group][var_name]
    return os.environ.get(var_name)