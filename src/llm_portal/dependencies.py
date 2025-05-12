import core
import message_broker
import utils

config = utils.get_config()


DEPENDENCIES = {
    "uow": core.UnitOfWork(config["database"]),
    "publisher": message_broker.Publisher(config["message_broker"]),
}
