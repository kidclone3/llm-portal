# import pydantic_settings
from core.configurations import DatabaseConfig
from yaml import safe_load

# class Settings(pydantic_settings.BaseSettings):
#     """Settings."""

#     model_config = pydantic_settings.SettingsConfigDict(
#         env_file=".env",
#         extra="ignore",
#         str_strip_whitespace=True,
#         validate_assignment=True,
#     )

#     DATABASE = DatabaseConfig()


# settings = Settings()


with open(".configs/database.yaml") as file:
    config = safe_load(file)
