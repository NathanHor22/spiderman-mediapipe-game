"""Primary Panda3D presentation layer for the swing simulation.

The package stays import-light so asset tools can use
``spidergame.render3d.buildings`` without importing Pygame or Panda3D.  Game
symbols are loaded only when a caller asks for them.
"""

from __future__ import annotations

from typing import Any


_GAME_EXPORTS = {
    "GameConfig",
    "GameStartupError",
    "GameState",
    "SpiderGame3D",
    "main",
}

__all__ = sorted(_GAME_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _GAME_EXPORTS:
        raise AttributeError(name)
    from . import game

    value = getattr(game, name)
    globals()[name] = value
    return value
