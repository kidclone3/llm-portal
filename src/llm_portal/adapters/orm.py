import core
import sqlalchemy
import utils
from core import orm
from core.adapters import sqlalchemy_adapter

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Enum,
    Float,
    Integer,
    String,
    Table,
    Text,
    event,
    types,
)

from llm_portal.domains import models


def setup_model_on_callbacks():
    def set_in_memory_attributes(obj: core.BaseModel, _):
        obj.events = []
        obj._immutable_atributes = {"id", "created_time"}

    for model in [
        models.EmbeddedResult,
    ]:
        event.listen(model, "load", set_in_memory_attributes)

@orm.map_once
def start_mapper():
    config_path = utils.get_config_path()
    config = utils.load_config(config_path=config_path)
    factory = sqlalchemy_adapter.ComponentFactory(config["database"])
    metadata = sqlalchemy.MetaData()
    orm_registry = sqlalchemy.orm.registry(metadata=metadata)

    product_table = Table(
        "embedded_results",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("text", Text, nullable=False),
        Column("provider", String(32), nullable=False),
        Column("model", String(64), nullable=False),
        Column("dimensions", Integer, nullable=False),
        Column("vector", JSON, nullable=False),

        Column("created_time", sqlalchemy.DateTime),
        Column("updated_time", sqlalchemy.DateTime),
    )
    embedded_mapper = orm_registry.map_imperatively(
        class_=models.EmbeddedResult,
        local_table=product_table,
    )
    engine = factory.engine
    setup_model_on_callbacks()
    metadata.create_all(engine)