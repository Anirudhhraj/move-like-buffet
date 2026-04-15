# Lazy package — nothing imported at module level.
# This prevents circular imports when sync_pipeline.py
# touches rag.indexer without needing the full agent chain.


def __getattr__(name):
    if name == "BuffettAgent":
        from .agent import BuffettAgent
        return BuffettAgent
    raise AttributeError(f"module 'rag' has no attribute {name}")


__all__ = ["BuffettAgent"]