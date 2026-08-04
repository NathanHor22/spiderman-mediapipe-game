"""Shared pygame UI: camera frames as surfaces, and HUD widgets.

Everything the project draws goes through pygame, including the diagnostic
tools. `cv2.imshow` was fine for a throwaway probe but it has no sound, no
proper event handling and its own frame pacing, so the tools would drift away
from how the game actually behaves — which defeats the point of a tool you use
to decide whether the game will feel right.
"""

from .surface import bgr_to_surface, fit_rect
from .hud import Panel, draw_bar, draw_text

__all__ = ["bgr_to_surface", "fit_rect", "Panel", "draw_bar", "draw_text"]
