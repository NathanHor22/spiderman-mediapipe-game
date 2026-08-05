"""Launch the Panda3D version of the endless swinger directly.

The normal ``run_game.py`` launcher now selects this renderer by default. This
entry point remains as a convenient direct alias; the earlier Pygame renderer
is available through ``run_game.py --legacy-renderer``.
"""

from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Panda3D game and return a process exit status."""
    from spidergame.render3d.game import main as game_main

    return game_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
