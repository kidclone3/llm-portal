import core

from llm_portal import dependencies
from llm_portal.service.handlers import command, event

BOOTSTRAPPER = core.Bootstrapper(
    use_orm=False,
    orm_func=lambda: None,
    command_router=command.COMMAND_HANDLERS,
    event_router=event.EVENT_HANDLERS,
    dependencies=dependencies.DEPENDENCIES,
)