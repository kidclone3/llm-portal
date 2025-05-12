import core

BOOTSTRAPPER = core.Bootstrapper(
    use_orm=False,
    orm_func=lambda: None,
    command_router=command.COMMAND_HANDLERS,
    event_router=event.EVENT_HANDLERS,
    dependencies=dependencies.DEPENDENCIES,
)