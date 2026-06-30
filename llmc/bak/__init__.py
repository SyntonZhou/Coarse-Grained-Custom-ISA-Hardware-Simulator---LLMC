"""llmc -- a tiny instruction-scheduling compiler for the edge-LLM accelerator."""
from . import isa, lowering  # noqa: F401

__all__ = ["isa", "lowering"]
__version__ = "0.1.0"
