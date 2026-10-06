from models.surrogate import GWFlowSurrogateModel

__all__ = ["GWFlowSurrogateLit", "GWFlowSurrogateModel"]


def __getattr__(name: str):
    if name == "GWFlowSurrogateLit":
        from models.lightning_module import GWFlowSurrogateLit
        return GWFlowSurrogateLit
    raise AttributeError(name)
