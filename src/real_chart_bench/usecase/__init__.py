"""Use case layer: application services orchestrating domain logic via port interfaces.

May import ``domain``. Must not import ``adapter`` or ``infrastructure`` concrete
classes — only depend on port interfaces defined here or in ``domain``.
"""
