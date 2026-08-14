"""Domain layer: entities, value objects, and pure evaluation logic.

Must not import from ``usecase``, ``adapter``, or ``infrastructure``, and must not
depend on any I/O (network, filesystem, CLI frameworks). This is enforced by the
``domain-has-no-outward-dependencies`` contract in the import-linter config.

Phase 1 adds ``Curve``, ``MetricStrategy``, and ``CurveMatcher`` here per
docs/design/benchmark-architecture.md §4.2.
"""
