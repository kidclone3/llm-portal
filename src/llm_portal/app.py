import logging

from llm_portal.entrypoints.rest import app as rest_app
import utils

logger = utils.get_logger()

def main():
    try:
        logger.info("Init application")
        rest_app.run()
    finally:
        logger.info("Application shutdown gracefully")

if __name__ == "__main__":
    main()