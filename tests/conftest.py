import os
import pathlib
from typing import Generator, Any

import pytest
import utils
from fastapi import testclient

from llm_portal.entrypoints.rest import app

logger = utils.get_logger()
config = utils.get_config()

@pytest.fixture(scope="session")
def project_path() -> Generator[pathlib.Path, Any, None]:
    yield pathlib.Path(__file__).parents[1]


@pytest.fixture(scope="class")
def config_path(project_path: pathlib.Path) -> str:
    return str(project_path / ".configs")


@pytest.fixture
def config(
    project_path: pathlib.Path,
    config_path: str,
) -> Generator[dict[str, Any], Any, None]:
    os.environ["CONFIG_PATH"] = config_path
    os.environ["ENVIRONMENT"] = "test"
    config_dict = utils.load_config(config_path)
    yield config_dict

@pytest.fixture
def cleanup() -> Generator[dict[str, bool], Any, None]:
    yield {
        "before": True,
        "after": False,
    }

@pytest.fixture
def rest_client(
        config: dict[str, Any],
        cleanup: dict[str, bool],
) -> Generator[testclient.TestClient, Any, None]:
    if cleanup["before"]:
        logger.info("Cleaning up before starting the test")
    yield testclient.TestClient(app.create_app())

    if cleanup["after"]:
        logger.info("Cleaning up after the test")