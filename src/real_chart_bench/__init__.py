"""real-chart-bench: an open benchmark for chart data extraction from real research figures.

Package layout follows clean architecture (see docs/design/benchmark-architecture.md §4):

- ``domain``: entities and pure logic (metrics, matching). No outward imports.
- ``usecase``: application services, depends only on domain + port interfaces.
- ``adapter``: implementations of ports (model runners, repositories).
- ``infrastructure``: CLI, HTTP, filesystem — the outermost, most volatile layer.

Dependency direction is always outward-to-inward:
infrastructure -> adapter -> usecase -> domain. Enforced in CI via import-linter
(see ``.importlinter`` / ``pyproject.toml``).
"""

__version__ = "0.0.1"
