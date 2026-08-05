"""Launch the optional Panda3D version of the endless swinger.

The existing :mod:`run_game` Pygame runner remains the default game.  This
entry point is intentionally separate while the real-time 3D asset pipeline is
being developed.
"""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Panda3D game and return a process exit status."""
    from spidergame.render3d.game import main as game_main

    return game_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
