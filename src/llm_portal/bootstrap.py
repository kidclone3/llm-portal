import core

from llm_portal import dependencies
from llm_portal.adapters import orm
from llm_portal.service.handlers import command, event

BOOTSTRAPPER = None

def bootstrap(use_orm: bool = True) -> core.MessageBus:
    global BOOTSTRAPPER
    if BOOTSTRAPPER is None:
        BOOTSTRAPPER = core.Bootstrapper(
            use_orm=use_orm,
            orm_func=orm.start_mapper,
            command_router=command.COMMAND_HANDLERS,
            event_router=event.EVENT_HANDLERS,
            dependencies=dependencies.DEPENDENCIES,
        )
    return BOOTSTRAPPER.bootstrap()