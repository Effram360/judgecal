"""judgecal.integrations — adapters for third-party evaluation harnesses.

Integration modules carry optional dependencies and are deliberately NOT
imported here: importing :mod:`judgecal.integrations` must always succeed,
even when no extras are installed. Import the specific integration you
need, e.g. ``from judgecal.integrations import inspect_ai`` (requires
``pip install 'judgecal[inspect]'``).
"""

from __future__ import annotations

__all__: list[str] = []
