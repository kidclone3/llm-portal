import fastapi
import utils

from llm_portal.domains import commands, models
from llm_portal import bootstrap
from llm_portal.entrypoints import schemas
from llm_portal.service import view

bus = bootstrap.bootstrap()
logger = utils.get_logger()
router = fastapi.APIRouter()



@router.post("/embedding", status_code=fastapi.status.HTTP_200_OK)
async def embedding(
        command: commands.InputTextCommand
)-> schemas.EmbeddedResponse:
    """
    Endpoint to get the embedding of a given text.

    Args:
        command (commands.InputTextCommand): The command containing text to embed.

    Returns:
    """
    # Placeholder for actual embedding logic
    # In a real implementation, this would call the appropriate service
    # to get the embedding based on the provider and model.
    try:
        bus.handle(command)

        with view.fetch_model(
            model_cls=models.EmbeddedResult,
            id=command._id,
        ) as embedded_result:
            return schemas.EmbeddedResponse(
                result=embedded_result,
            )
    except Exception as e:
        logger.error(e)
        raise fastapi.HTTPException(
            status_code=fastapi.status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

