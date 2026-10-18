from .utils import dump_graph

__all__ = [
    "ExportedCapture",
    "TransformerCapture",
    "capture_exported_program",
    "capture_transformer",
    "dump_graph",
    "register_transformers_attention",
    "DecodeStep",
]


def __getattr__(name: str):
    if name in {"ExportedCapture", "capture_exported_program"}:
        from . import export

        return getattr(export, name)
    if name in {"register_transformers_attention", "DecodeStep"}:
        from . import hf_attention

        return getattr(hf_attention, name)
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
