import logging
import os
import pathlib

import utils

project_path = pathlib.Path(__file__).parents[2]

os.environ["PROJECT_PATH"] = str(project_path)
os.environ["CONFIG_PATH"] = os.environ.get("CONFIG_PATH", str(project_path / ".configs"))

config = utils.load_config(os.environ.get("CONFIG_PATH"))
logger = utils.get_logger()
logger.setLevel(logging.DEBUG)