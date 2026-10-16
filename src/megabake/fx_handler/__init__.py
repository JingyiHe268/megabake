from .utils import dump_graph

__all__ = [
    "ExportedCapture",
    "TransformerCapture",
    "capture_exported_program",
    "capture_transformer",
    "dump_graph",
]


def __getattr__(name: str):
    if name in {"ExportedCapture", "capture_exported_program"}:
        from . import export

        return getattr(export, name)
    if name in {"TransformerCapture", "capture_transformer"}:
        from . import transformers

        return getattr(transformers, name)
    raise AttributeError(name)

__all__ = [
    "ExportedCapture",
    "TransformerCapture",
    "capture_exported_program",
    "capture_transformer",
]
