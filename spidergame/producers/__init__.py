"""ControlState producers. Swappable — the game cannot tell them apart."""

from .keyboard import KeyboardProducer
from .vision import VisionProducer

__all__ = ["KeyboardProducer", "VisionProducer"]
