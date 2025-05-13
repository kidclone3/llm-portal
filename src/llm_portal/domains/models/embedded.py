import core


class EmbeddedResult(core.BaseModel):
    def __init__(self,
                 id: str,
                 text: str, provider: str, model: str, dimensions: int, vector: list[float], *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.id = id
        self.text = text
        self.provider = provider
        self.model = model
        self.dimensions = dimensions
        self.vector = vector