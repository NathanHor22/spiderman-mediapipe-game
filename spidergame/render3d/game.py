"""Panda3D game loop backed by the existing swing simulation.

The module keeps Panda3D optional: pure configuration helpers can be imported
on machines that only run the Pygame version, and launching this runner without
Panda3D produces an actionable error instead of an import traceback.

Simulation coordinates are ``(x lateral, y altitude, z forward)``.  Panda3D
uses Z-up coordinates, so every presentation object maps them to
``(x, z, y)``.  Physics remains authoritative; rendering never feeds positions
back into :class:`~spidergame.game.swing.SwingSim`.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

import pygame

from spidergame.audio import SoundSystem, prepare_mixer
from spidergame.control import ControlState
from spidergame.game import tuning as T
from spidergame.game.swing import SwingSim
from spidergame.producers.keyboard import KeyboardProducer
from spidergame.render.world import STREET_HALF, WorldStrip
from spidergame.render3d.settings import (
    SettingsIntent,
    SettingsMenu,
    SettingsRow,
)
from spidergame.render3d.tutorial import (
    Countdown,
    MenuIntent,
    TitleMenu,
    TutorialController,
    TutorialStep,
)


_PANDA_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from direct.gui.DirectGui import DGG, DirectButton, DirectFrame, DirectWaitBar
    from direct.gui.OnscreenImage import OnscreenImage
    from direct.gui.OnscreenText import OnscreenText
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        AmbientLight,
        CardMaker,
        ClockObject,
        DirectionalLight,
        Fog,
        KeyboardButton,
        LineSegs,
        MouseButton,
        TextNode,
        Texture,
        TransparencyAttrib,
        Vec3,
        loadPrcFileData,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    if exc.name not in {"direct", "panda3d"} and not (
        exc.name and exc.name.startswith(("direct.", "panda3d."))
    ):
        raise
    _PANDA_IMPORT_ERROR = exc
    ShowBase = object  # type: ignore[assignment,misc]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUILDING_ASSET = (
    PROJECT_ROOT / "assets" / "models" / "buildings" / "new_york_generic.glb"
)
DEFAULT_BUILDING_MANIFEST = (
    PROJECT_ROOT / "assets" / "models" / "buildings" / "manifest.json"
)
DEFAULT_CHARACTER_ASSET = (
    PROJECT_ROOT / "assets" / "models" / "character" / "spider_man.glb"
)
DEFAULT_CHARACTER_MANIFEST = (
    PROJECT_ROOT / "assets" / "models" / "character" / "character_manifest.json"
)
DEFAULT_THWIP_IMAGE = PROJECT_ROOT / "assets" / "ui" / "spider-man-thwip.png"

PLAYER_HEIGHT = 4.8
CAMERA_BACK = 18.0
CAMERA_UP = 6.2
CAMERA_LOOK_AHEAD = 7.0
CAMERA_RESPONSE = 7.5
ROAD_LENGTH = 1_200.0
MAX_FRAME_DT = 0.05
LOADING_MIN_SECONDS = 0.65
VISION_STARTUP_TIMEOUT = 45.0
VISION_SWITCH_TIMEOUT = 10.0
CAMERA_SOURCE_COUNT = 4


class GameStartupError(RuntimeError):
    """A dependency, window, or required building asset cannot be prepared."""


class GameState(Enum):
    """Small state machine shared by keyboard callbacks and the frame task."""

    LOADING = auto()
    TITLE = auto()
    TUTORIAL = auto()
    SETTINGS = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    DEAD = auto()


@dataclass(frozen=True)
class GameConfig:
    """Resolved settings for one Panda3D process."""

    seed: int = 11
    building_asset: Path = DEFAULT_BUILDING_ASSET
    building_manifest: Path = DEFAULT_BUILDING_MANIFEST
    character_asset: Path | None = DEFAULT_CHARACTER_ASSET
    character_manifest: Path | None = None
    audio: bool = True
    vision: bool = True
    camera_index: int | None = None
    skip_title: bool = False
    headless: bool = False
    max_frames: int | None = None


def simulation_to_render(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Map simulation axes to Panda3D's X/Y/Z convention."""

    return float(x), float(z), float(y)


def velocity_to_render_hpr(
    vx: float,
    vy: float,
    vz: float,
) -> tuple[float, float, float]:
    """Face along travel and bank the torso with the vertical flight path."""

    heading = -math.degrees(math.atan2(vx, max(1e-6, vz)))
    horizontal_speed = math.hypot(vx, vz)
    pitch = math.degrees(math.atan2(vy, max(1e-6, horizontal_speed)))
    return heading, max(-18.0, min(18.0, pitch)), 0.0


def _resolved_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _short_camera_message(value: object, limit: int = 82) -> str:
    """Keep driver diagnostics inside the fixed-width camera card."""

    message = " ".join(str(value).split())
    if len(message) <= limit:
        return message
    return message[: max(1, limit - 3)].rstrip() + "..."


def _camera_switch_allowed(state: GameState, vision: bool) -> bool:
    """Camera controls are useful anywhere the live preview is visible."""

    return bool(vision) and state in {
        GameState.TITLE,
        GameState.TUTORIAL,
        GameState.SETTINGS,
    }


def validate_building_assets(asset_path: Path, manifest_path: Path) -> None:
    """Fail before opening a window when required city data is incomplete."""

    missing: list[str] = []
    if not asset_path.is_file():
        missing.append(f"building model: {asset_path}")
    if not manifest_path.is_file():
        missing.append(f"building manifest: {manifest_path}")
    if missing:
        detail = "\n  - ".join(missing)
        raise GameStartupError(
            "Panda3D building assets are incomplete. Missing:\n"
            f"  - {detail}\n"
            "Export the city assets there, or pass --building-asset and "
            "--building-manifest."
        )


def require_panda3d() -> None:
    """Raise an installation hint without leaking an optional-import traceback."""

    if _PANDA_IMPORT_ERROR is not None:
        raise GameStartupError(
            "Panda3D is required for the 3D game. Install it with "
            "`python -m pip install panda3d`, then run this entry point again."
        ) from _PANDA_IMPORT_ERROR


class PandaKeyboardProducer(KeyboardProducer):
    """Adapt Panda's window state to the existing keyboard control producer.

    ``KeyboardProducer`` remains the input seam and owns punch edge latching;
    only its Pygame window polling is replaced because Panda3D owns this window.
    """

    def __init__(self, base: Any) -> None:
        super().__init__()
        self._base = base

    def latch_punch(self) -> None:
        """Feed a Panda key edge through ``KeyboardProducer``'s latch."""

        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f)
        self.handle_event(event)

    def poll(self) -> ControlState:
        watcher = getattr(self._base, "mouseWatcherNode", None)
        window = getattr(self._base, "win", None)
        thwip = False
        two_hands = False
        hand_x = hand_y = 0.5

        if watcher is not None:
            thwip = bool(
                watcher.isButtonDown(KeyboardButton.space())
                or watcher.isButtonDown(MouseButton.one())
            )
            two_hands = bool(watcher.isButtonDown(KeyboardButton.shift()))

        has_pointer = getattr(window, "hasPointer", None)
        get_pointer = getattr(window, "getPointer", None)
        if (
            callable(has_pointer)
            and callable(get_pointer)
            and has_pointer(0)
        ):
            pointer = get_pointer(0)
            width = max(1, window.getXSize())
            height = max(1, window.getYSize())
            hand_x = min(max(pointer.getX() / width, 0.0), 1.0)
            hand_y = min(max(pointer.getY() / height, 0.0), 1.0)

        fired = self._punch_latched
        self._punch_latched = False
        return ControlState(
            thwip_held=thwip,
            hand_x=hand_x,
            hand_y=hand_y,
            num_hands=2 if two_hands else 1,
            punch_fired=fired,
            tracking_lost=False,
        )


class _SilentSoundSystem:
    """SoundSystem-compatible object used by --no-audio and headless runs."""

    enabled = False
    error = "audio disabled"
    music_playing = False
    loaded_assets: tuple[str, ...] = ()

    def handle(self, _events: Any, _sim: Any = None) -> None:
        pass

    def update(self, _sim: Any) -> None:
        pass

    def play_fall(self) -> None:
        pass

    def play_menu_music(self, fade_ms: int = 900) -> None:
        del fade_ms

    def stop_music(self, fade_ms: int = 600) -> None:
        del fade_ms

    def stop(self, fade_ms: int = 80) -> None:
        del fade_ms

    def close(self) -> None:
        pass


class SpiderGame3D(ShowBase):  # type: ignore[misc]
    """Close third-person Panda3D presentation of the endless swing game."""

    def __init__(self, config: GameConfig) -> None:
        require_panda3d()
        validate_building_assets(config.building_asset, config.building_manifest)
        # ShowBase owns ``self.config`` for Panda's ConfigPageManager.  Keep
        # game settings under a distinct name or window creation fails before
        # the graphics pipe is selected.
        self.game_config = config

        prc = [
            "window-title Spider - Panda3D Swing",
            "win-size 1280 720",
            "sync-video 0",
            "show-frame-rate-meter 0",
        ]
        if config.headless:
            prc.extend(("window-type offscreen", "audio-library-name null"))
        loadPrcFileData("spidergame-render3d", "\n".join(prc))

        try:
            super().__init__(windowType="offscreen" if config.headless else None)
        except Exception as exc:  # Panda reports graphics-pipe failures broadly.
            raise GameStartupError(
                "Panda3D could not open a rendering window. Check the graphics "
                f"driver or use --headless for a smoke test: {exc}"
            ) from exc

        self.disableMouse()
        self.setBackgroundColor(0.025, 0.035, 0.075, 1.0)
        self.camLens.setNearFar(0.15, 620.0)
        self.camLens.setFov(76.0)
        self.clock = ClockObject.getGlobalClock()

        self._closed = False
        self._frames = 0
        self._title_distance = 0.0
        self._camera_ready = False
        self._loading_elapsed = 0.0
        self._vision_deadline = 0.0
        self._vision_ready = not config.vision
        self._camera_has_frame = False
        self._camera_frame_token: int | None = None
        self._camera_texture_size: tuple[int, int] | None = None
        self.camera_index = config.camera_index
        self.auto_picked_camera = False
        self.vision_error: str | None = None
        self.camera_notice: str | None = None
        self._camera_notice_until = 0.0
        self.control = ControlState(tracking_lost=config.vision)
        self.best_distance = 0.0
        self.title_menu = TitleMenu()
        self.tutorial = TutorialController(vision=config.vision)
        self.countdown = Countdown()
        self.settings_menu: SettingsMenu | None = None
        self.settings_message = ""
        self.producer: Any | None = None

        self._setup_boot_ui()
        self._render_boot_stage("PREPARING THE CITY", 8.0)
        self._setup_lighting_and_fog()
        self._setup_road()
        self.player = self.render.attachNewNode("player-root")
        self._setup_character_lighting()
        self.character_controller = None
        self._render_boot_stage("LOADING THE ANIMATED HERO", 28.0)
        self.character_note = self._load_character(
            config.character_asset,
            config.character_manifest,
        )
        self._web_node = None

        self._render_boot_stage("ASSEMBLING NEW YORK", 48.0)
        self.building_renderer = self._create_building_renderer()
        self._render_boot_stage("PREPARING SWING SOUND", 66.0)
        self.audio = self._create_audio()

        self.world = WorldStrip(seed=config.seed)
        self.backdrop_world = WorldStrip(seed=config.seed + 7)
        self.sim = SwingSim()
        self.world.update(self.sim.z + 60.0)
        self.backdrop_world.update(60.0)
        self._sync_buildings(self.backdrop_world)

        self.state = GameState.LOADING
        self._clear_boot_ui()
        self._setup_ui()
        self._set_state(GameState.LOADING)
        self._render_loading_stage("CITY AND HERO READY", 72.0)
        self.producer = self._create_producer()
        self.settings_menu = self._create_settings_menu()
        self._bind_controls()
        self.taskMgr.add(self._update, "spidergame-3d-update", sort=10)

    # --------------------------------------------------------------- startup

    def _setup_boot_ui(self) -> None:
        """Create a minimal overlay before any model or audio work begins."""

        common = dict(parent=self.aspect2d, mayChange=True)
        self._boot_nodes = [
            DirectFrame(
                parent=self.aspect2d,
                frameSize=(-1.78, 1.78, -1.0, 1.0),
                frameColor=(0.015, 0.020, 0.050, 1.0),
            ),
            OnscreenText(
                text="SPIDER SWING",
                pos=(0.0, 0.26),
                scale=0.14,
                align=TextNode.ACenter,
                fg=(0.94, 0.96, 1.0, 1.0),
                shadow=(0.02, 0.02, 0.05, 0.95),
                **common,
            ),
        ]
        self._boot_text = OnscreenText(
            text="STARTING",
            pos=(0.0, -0.02),
            scale=0.052,
            align=TextNode.ACenter,
            fg=(0.72, 0.80, 0.94, 1.0),
            **common,
        )
        self._boot_bar = DirectWaitBar(
            parent=self.aspect2d,
            pos=(0.0, 0.0, -0.20),
            frameSize=(-0.52, 0.52, -0.025, 0.025),
            frameColor=(0.05, 0.07, 0.13, 1.0),
            barColor=(0.80, 0.08, 0.12, 1.0),
            relief=DGG.FLAT,
            range=100,
            value=0,
            text="",
        )
        self._boot_hint = OnscreenText(
            text="Loading real buildings, character and animation...",
            pos=(0.0, -0.31),
            scale=0.035,
            align=TextNode.ACenter,
            fg=(0.48, 0.56, 0.70, 1.0),
            **common,
        )
        self._boot_nodes.extend(
            (self._boot_text, self._boot_bar, self._boot_hint)
        )

    def _render_boot_stage(self, message: str, progress: float) -> None:
        self._boot_text.setText(message)
        self._boot_bar["value"] = max(0.0, min(100.0, float(progress)))
        try:
            self.graphicsEngine.renderFrame()
        except Exception:
            pass

    def _clear_boot_ui(self) -> None:
        for node in self._boot_nodes:
            destroy = getattr(node, "destroy", None)
            if callable(destroy):
                destroy()
            else:
                node.removeNode()
        self._boot_nodes = []

    def _create_building_renderer(self) -> Any:
        try:
            from spidergame.render3d.buildings import BuildingRenderer
        except (ImportError, ModuleNotFoundError) as exc:
            raise GameStartupError(
                "The Panda3D building renderer is unavailable at "
                "spidergame.render3d.buildings."
            ) from exc

        try:
            return BuildingRenderer(
                self.render,
                self.game_config.building_asset,
                self.game_config.building_manifest,
            )
        except Exception as exc:
            raise GameStartupError(
                "Could not initialise the Panda3D building assets "
                f"({self.game_config.building_asset}): {exc}"
            ) from exc

    def _create_audio(self) -> SoundSystem | _SilentSoundSystem:
        if not self.game_config.audio or self.game_config.headless:
            return _SilentSoundSystem()
        prepare_mixer()
        audio = SoundSystem()
        if not audio.enabled and audio.error:
            print(
                f"3D runner: audio unavailable; continuing silently: {audio.error}",
                file=sys.stderr,
            )
        return audio

    def _create_producer(self) -> Any:
        if not self.game_config.vision:
            self._vision_ready = True
            self._render_loading_stage("KEYBOARD AND MOUSE READY", 88.0)
            return PandaKeyboardProducer(self)

        from spidergame.producers.vision import VisionProducer

        index = self.game_config.camera_index
        if index is None:
            self._render_loading_stage("LOOKING FOR A WORKING CAMERA", 78.0)
            from spidergame.vision import devices

            index = devices.pick_default()
            if index is None:
                index = 0
                self.vision_error = (
                    "no camera delivered usable pixels; check the privacy "
                    "shutter and Windows camera permissions"
                )
            else:
                self.auto_picked_camera = True
                self.camera_notice = f"camera {index} auto-selected"
                self._camera_notice_until = time.monotonic() + 4.0
        self.camera_index = index
        self._render_loading_stage(
            f"CONNECTING TO CAMERA {index}\nFIRST RUN MAY DOWNLOAD THE HAND MODEL",
            86.0,
        )
        producer = VisionProducer(
            camera_index=index,
            keep_frame=True,
        )
        self._vision_ready = False
        self._vision_deadline = time.monotonic() + VISION_STARTUP_TIMEOUT
        return producer

    def _create_settings_menu(self) -> SettingsMenu:
        """Build predictable capture-index choices without reopening devices."""

        if not self.game_config.vision:
            return SettingsMenu([])
        active = 0 if self.camera_index is None else int(self.camera_index)
        upper = max(CAMERA_SOURCE_COUNT - 1, active)
        return SettingsMenu(range(upper + 1), active_camera=active)

    def _render_loading_stage(self, message: str, progress: float) -> None:
        """Publish a startup stage before a potentially blocking probe."""

        if not hasattr(self, "loading_text"):
            return
        self.loading_text.setText(message)
        self.loading_bar["value"] = max(0.0, min(100.0, float(progress)))
        try:
            self.graphicsEngine.renderFrame()
        except Exception:
            # An offscreen smoke test may not have a complete output yet.
            pass

    def _setup_lighting_and_fog(self) -> None:
        ambient = AmbientLight("ambient")
        ambient.setColor((0.34, 0.38, 0.50, 1.0))
        ambient_np = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_np)

        sun = DirectionalLight("moon-key")
        sun.setColor((0.82, 0.86, 1.0, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-35.0, -52.0, 0.0)
        self.render.setLight(sun_np)

        fog = Fog("city-haze")
        fog.setColor(0.055, 0.070, 0.13)
        fog.setLinearRange(125.0, 500.0)
        self.render.setFog(fog)

    def _setup_character_lighting(self) -> None:
        """Keep the suit readable without flattening the night-time city."""

        fill = AmbientLight("character-fill")
        fill.setColor((0.16, 0.17, 0.21, 1.0))
        self._character_fill = self.render.attachNewNode(fill)
        self.player.setLight(self._character_fill)

    def _setup_road(self) -> None:
        card = CardMaker("street")
        card.setFrame(-STREET_HALF, STREET_HALF, -ROAD_LENGTH / 2, ROAD_LENGTH / 2)
        self.road = self.render.attachNewNode(card.generate())
        self.road.setP(-90.0)
        self.road.setZ(0.02)
        self.road.setColor(0.055, 0.060, 0.075, 1.0)
        self.road.setTwoSided(True)

    def _load_character(
        self,
        asset_path: Path | None,
        manifest_path: Path | None,
    ) -> str:
        if asset_path is not None and asset_path.is_file():
            try:
                from spidergame.render3d.character import CharacterController

                self.character_controller = CharacterController(
                    self.player,
                    asset_path,
                    target_height=PLAYER_HEIGHT,
                    manifest_path=manifest_path,
                )
                return f"character: animated {asset_path.name}"
            except Exception as exc:
                print(
                    f"3D runner: could not load character {asset_path}; "
                    f"using placeholder: {exc}",
                    file=sys.stderr,
                )

        self._build_placeholder_character()
        if asset_path is None:
            return "character: tall placeholder"
        if not asset_path.is_file():
            print(
                f"3D runner: character model not found at {asset_path}; "
                "using a tall placeholder.",
                file=sys.stderr,
            )
        return "character: tall placeholder"

    def _build_placeholder_character(self) -> None:
        """Create an asset-free, 4.8-unit stick figure around the sim point."""

        line = LineSegs("placeholder-spider")
        line.setThickness(7.0)
        line.setColor(0.82, 0.035, 0.055, 1.0)

        # Torso, arms, neck and head in a forward-facing vertical plane.
        line.moveTo(0.0, 0.0, -0.8)
        line.drawTo(0.0, 0.0, 1.25)
        line.moveTo(-1.25, 0.0, 0.45)
        line.drawTo(1.25, 0.0, 0.45)
        line.moveTo(0.0, 0.0, 1.25)
        line.drawTo(0.0, 0.0, 1.55)
        head_radius = 0.43
        for index in range(17):
            angle = math.tau * index / 16
            point = (
                math.cos(angle) * head_radius,
                0.0,
                1.92 + math.sin(angle) * head_radius,
            )
            if index == 0:
                line.moveTo(*point)
            else:
                line.drawTo(*point)

        line.setColor(0.06, 0.18, 0.58, 1.0)
        line.moveTo(0.0, 0.0, -0.8)
        line.drawTo(-0.68, 0.0, -2.35)
        line.moveTo(0.0, 0.0, -0.8)
        line.drawTo(0.68, 0.0, -2.35)
        self.player.attachNewNode(line.create())

    def _setup_ui(self) -> None:
        common = dict(parent=self.aspect2d, mayChange=True)
        menu_x = -0.68 if self.game_config.vision else 0.0
        camera_x = 0.78
        tutorial_x = -0.68 if self.game_config.vision else 0.0
        settings_x = -0.68 if self.game_config.vision else 0.0
        self.loading_panel = DirectFrame(
            parent=self.aspect2d,
            frameSize=(-1.78, 1.78, -1.0, 1.0),
            frameColor=(0.015, 0.020, 0.050, 0.92),
        )
        self.loading_title = OnscreenText(
            text="SPIDER SWING",
            pos=(0.0, 0.26),
            scale=0.14,
            align=TextNode.ACenter,
            fg=(0.94, 0.96, 1.0, 1.0),
            shadow=(0.02, 0.02, 0.05, 0.95),
            **common,
        )
        self.loading_text = OnscreenText(
            text="LOADING CITY",
            pos=(0.0, -0.02),
            scale=0.052,
            align=TextNode.ACenter,
            fg=(0.72, 0.80, 0.94, 1.0),
            **common,
        )
        self.loading_bar = DirectWaitBar(
            parent=self.aspect2d,
            pos=(0.0, 0.0, -0.20),
            frameSize=(-0.52, 0.52, -0.025, 0.025),
            frameColor=(0.05, 0.07, 0.13, 1.0),
            barColor=(0.80, 0.08, 0.12, 1.0),
            relief=DGG.FLAT,
            range=100,
            value=8,
            text="",
        )
        self.loading_hint = OnscreenText(
            text="Preparing the animated city and gesture controls...",
            pos=(0.0, -0.31),
            scale=0.035,
            align=TextNode.ACenter,
            fg=(0.48, 0.56, 0.70, 1.0),
            **common,
        )

        self.title_panel = DirectFrame(
            parent=self.aspect2d,
            pos=(menu_x, 0.0, 0.0),
            frameSize=(-0.66, 0.66, -0.79, 0.79),
            frameColor=(0.018, 0.024, 0.060, 0.92),
        )
        self.title_text = OnscreenText(
            text="SPIDER SWING",
            pos=(menu_x, 0.67),
            scale=0.105,
            align=TextNode.ACenter,
            fg=(0.96, 0.12, 0.15, 1.0),
            shadow=(0.02, 0.02, 0.04, 0.9),
            **common,
        )
        self.title_subtitle = OnscreenText(
            text="ENDLESS CITY SWINGER  //  THIRD PERSON",
            pos=(menu_x, 0.53),
            scale=0.032,
            align=TextNode.ACenter,
            fg=(0.55, 0.67, 0.95, 1.0),
            **common,
        )
        self.title_buttons: list[Any] = []
        for index, label in enumerate(self.title_menu.labels):
            button = DirectButton(
                parent=self.aspect2d,
                text=label,
                pos=(menu_x, 0.0, 0.29 - index * 0.13),
                scale=0.050,
                frameSize=(-9.4, 9.4, -0.82, 0.82),
                frameColor=(0.05, 0.07, 0.14, 0.96),
                text_fg=(0.82, 0.87, 0.98, 1.0),
                text_shadow=(0.01, 0.01, 0.02, 0.9),
                relief=DGG.FLAT,
                rolloverSound=None,
                clickSound=None,
                command=self._activate_title_button,
                extraArgs=[index],
            )
            button.bind(DGG.ENTER, self._hover_title_button, [index])
            self.title_buttons.append(button)
        control_help = (
            "AIM       Move your hand left or right\n"
            "THWIP     Index + pinky out; curl middle + ring\n"
            "RELEASE   Relax the sign near the top of the arc"
            if self.game_config.vision
            else (
                "AIM       Move the mouse left or right\n"
                "THWIP     Hold SPACE or left mouse\n"
                "RELEASE   Let go near the top of the arc"
            )
        )
        self.title_help = OnscreenText(
            text=control_help,
            pos=(menu_x - 0.54, -0.31),
            scale=0.030,
            align=TextNode.ALeft,
            fg=(0.78, 0.84, 0.95, 1.0),
            shadow=(0.02, 0.02, 0.04, 0.9),
            wordwrap=36,
            **common,
        )
        self.title_nav_hint = OnscreenText(
            text="UP / DOWN  SELECT     ENTER  CONFIRM",
            pos=(menu_x, -0.57),
            scale=0.027,
            align=TextNode.ACenter,
            fg=(0.48, 0.57, 0.74, 1.0),
            **common,
        )
        hero_status = (
            "ANIMATED HERO"
            if self.character_controller is not None
            else "PLACEHOLDER HERO"
        )
        self.asset_text = OnscreenText(
            text=(
                f"{hero_status}  //  "
                f"AUDIO {'READY' if self.audio.enabled else 'OFF'}"
            ),
            pos=(menu_x, -0.71),
            scale=0.024,
            align=TextNode.ACenter,
            fg=(0.58, 0.64, 0.74, 1.0),
            **common,
        )

        self.camera_border = DirectFrame(
            parent=self.aspect2d,
            pos=(camera_x, 0.0, 0.02),
            frameSize=(-0.62, 0.62, -0.62, 0.66),
            frameColor=(0.018, 0.026, 0.062, 0.96),
            relief=DGG.FLAT,
        )
        self._camera_texture = Texture("live-camera-preview")
        self._camera_texture.setup2dTexture(
            2,
            2,
            Texture.T_unsigned_byte,
            Texture.F_rgb8,
        )
        self._camera_texture.setRamImage(bytes((12, 16, 30) * 4))
        self.camera_image = OnscreenImage(
            parent=self.aspect2d,
            image=self._camera_texture,
            pos=(camera_x, 0.0, 0.12),
            scale=(0.54, 1.0, 0.35),
        )
        self.camera_header = OnscreenText(
            text="LIVE CAMERA",
            pos=(camera_x, 0.57),
            scale=0.037,
            align=TextNode.ACenter,
            fg=(0.82, 0.88, 1.0, 1.0),
            **common,
        )
        self.camera_status = OnscreenText(
            text="CAMERA STARTING",
            pos=(camera_x, -0.31),
            scale=0.034,
            align=TextNode.ACenter,
            fg=(1.0, 0.72, 0.20, 1.0),
            **common,
        )
        self.camera_help = OnscreenText(
            text="Open SETTINGS to choose a camera source",
            pos=(camera_x, -0.43),
            scale=0.026,
            align=TextNode.ACenter,
            fg=(0.50, 0.58, 0.72, 1.0),
            wordwrap=43,
            **common,
        )

        self.tutorial_panel = DirectFrame(
            parent=self.aspect2d,
            pos=(tutorial_x, 0.0, 0.0),
            frameSize=(-0.72, 0.72, -0.74, 0.74),
            frameColor=(0.015, 0.022, 0.058, 0.94),
        )
        self.tutorial_counter = OnscreenText(
            text="TUTORIAL 1 / 3",
            pos=(tutorial_x - 0.61, 0.65),
            scale=0.029,
            align=TextNode.ALeft,
            fg=(0.54, 0.64, 0.83, 1.0),
            **common,
        )
        self.tutorial_title = OnscreenText(
            text="MAKE THE WEB-SHOOTER SIGN",
            pos=(tutorial_x, 0.53),
            scale=0.050,
            align=TextNode.ACenter,
            fg=(0.96, 0.97, 1.0, 1.0),
            shadow=(0.02, 0.02, 0.05, 0.9),
            **common,
        )
        self.tutorial_hint = OnscreenText(
            text="",
            pos=(tutorial_x, 0.40),
            scale=0.029,
            align=TextNode.ACenter,
            fg=(0.62, 0.70, 0.86, 1.0),
            wordwrap=42,
            **common,
        )
        self.tutorial_status = OnscreenText(
            text="MAKE THE SIGN",
            pos=(tutorial_x, -0.24),
            scale=0.039,
            align=TextNode.ACenter,
            fg=(1.0, 0.72, 0.20, 1.0),
            **common,
        )
        self.tutorial_prompt = OnscreenText(
            text="hold it...",
            pos=(tutorial_x, -0.31),
            scale=0.028,
            align=TextNode.ACenter,
            fg=(0.48, 0.56, 0.70, 1.0),
            **common,
        )
        self.tutorial_progress = DirectWaitBar(
            parent=self.aspect2d,
            pos=(tutorial_x, 0.0, -0.43),
            frameSize=(-0.51, 0.51, -0.019, 0.019),
            frameColor=(0.05, 0.07, 0.13, 1.0),
            barColor=(0.12, 0.82, 0.48, 1.0),
            relief=DGG.FLAT,
            range=100,
            value=0,
            text="",
        )
        self.tutorial_escape = OnscreenText(
            text="ESC  BACK TO TITLE",
            pos=(tutorial_x, -0.62),
            scale=0.026,
            align=TextNode.ACenter,
            fg=(0.48, 0.56, 0.70, 1.0),
            **common,
        )
        self.thwip_reference = self._build_thwip_reference(tutorial_x)

        self.settings_panel = DirectFrame(
            parent=self.aspect2d,
            pos=(settings_x, 0.0, 0.0),
            frameSize=(-0.72, 0.72, -0.74, 0.74),
            frameColor=(0.015, 0.022, 0.058, 0.95),
        )
        self.settings_title = OnscreenText(
            text="SETTINGS",
            pos=(settings_x, 0.64),
            scale=0.075,
            align=TextNode.ACenter,
            fg=(0.96, 0.97, 1.0, 1.0),
            shadow=(0.02, 0.02, 0.05, 0.9),
            **common,
        )
        self.settings_subtitle = OnscreenText(
            text="CAMERA AND INPUT",
            pos=(settings_x, 0.52),
            scale=0.030,
            align=TextNode.ACenter,
            fg=(0.55, 0.67, 0.95, 1.0),
            **common,
        )
        self.settings_source_heading = OnscreenText(
            text="CAMERA SOURCE",
            pos=(settings_x, 0.37),
            scale=0.031,
            align=TextNode.ACenter,
            fg=(0.64, 0.72, 0.88, 1.0),
            **common,
        )
        self.settings_source_frame = DirectFrame(
            parent=self.aspect2d,
            pos=(settings_x, 0.0, 0.16),
            frameSize=(-0.49, 0.49, -0.13, 0.13),
            frameColor=(0.045, 0.060, 0.125, 0.98),
            relief=DGG.FLAT,
        )
        self.settings_source_value = OnscreenText(
            text="CAMERA 0",
            pos=(settings_x, 0.19),
            scale=0.043,
            align=TextNode.ACenter,
            fg=(0.94, 0.96, 1.0, 1.0),
            **common,
        )
        self.settings_source_position = OnscreenText(
            text="1 OF 4",
            pos=(settings_x, 0.10),
            scale=0.024,
            align=TextNode.ACenter,
            fg=(0.48, 0.58, 0.76, 1.0),
            **common,
        )
        self.settings_prev_button = DirectButton(
            parent=self.aspect2d,
            text="<",
            pos=(settings_x - 0.59, 0.0, 0.16),
            frameSize=(-0.085, 0.085, -0.13, 0.13),
            frameColor=(0.08, 0.11, 0.22, 0.98),
            text_fg=(0.96, 0.97, 1.0, 1.0),
            text_scale=0.052,
            text_pos=(0.0, -0.018),
            relief=DGG.FLAT,
            rolloverSound=None,
            clickSound=None,
            command=self._cycle_settings_camera,
            extraArgs=[-1],
        )
        self.settings_next_button = DirectButton(
            parent=self.aspect2d,
            text=">",
            pos=(settings_x + 0.59, 0.0, 0.16),
            frameSize=(-0.085, 0.085, -0.13, 0.13),
            frameColor=(0.08, 0.11, 0.22, 0.98),
            text_fg=(0.96, 0.97, 1.0, 1.0),
            text_scale=0.052,
            text_pos=(0.0, -0.018),
            relief=DGG.FLAT,
            rolloverSound=None,
            clickSound=None,
            command=self._cycle_settings_camera,
            extraArgs=[1],
        )
        self.settings_source_hint = OnscreenText(
            text="Choose a source, then apply it to the live preview.",
            pos=(settings_x, -0.01),
            scale=0.027,
            align=TextNode.ACenter,
            fg=(0.55, 0.64, 0.80, 1.0),
            wordwrap=42,
            **common,
        )
        self.settings_apply_button = DirectButton(
            parent=self.aspect2d,
            text="APPLY CAMERA",
            pos=(settings_x, 0.0, -0.20),
            frameSize=(-0.52, 0.52, -0.058, 0.058),
            frameColor=(0.72, 0.04, 0.08, 0.98),
            text_fg=(1.0, 1.0, 1.0, 1.0),
            text_scale=0.040,
            text_pos=(0.0, -0.014),
            relief=DGG.FLAT,
            rolloverSound=None,
            clickSound=None,
            command=self._activate_settings_apply,
        )
        self.settings_back_button = DirectButton(
            parent=self.aspect2d,
            text="BACK TO MENU",
            pos=(settings_x, 0.0, -0.35),
            frameSize=(-0.52, 0.52, -0.058, 0.058),
            frameColor=(0.05, 0.07, 0.14, 0.98),
            text_fg=(0.76, 0.82, 0.94, 1.0),
            text_scale=0.040,
            text_pos=(0.0, -0.014),
            relief=DGG.FLAT,
            rolloverSound=None,
            clickSound=None,
            command=self._activate_settings_back,
        )
        self.settings_status = OnscreenText(
            text="",
            pos=(settings_x, -0.50),
            scale=0.028,
            align=TextNode.ACenter,
            fg=(0.55, 0.68, 0.90, 1.0),
            wordwrap=42,
            **common,
        )
        self.settings_footer = OnscreenText(
            text="ARROWS / WASD  NAVIGATE   //   ENTER  SELECT   //   ESC  BACK",
            pos=(settings_x, -0.65),
            scale=0.023,
            align=TextNode.ACenter,
            fg=(0.42, 0.50, 0.66, 1.0),
            **common,
        )
        self.settings_apply_button.bind(
            DGG.ENTER,
            self._hover_settings_row,
            [SettingsRow.APPLY],
        )
        self.settings_back_button.bind(
            DGG.ENTER,
            self._hover_settings_row,
            [SettingsRow.BACK],
        )
        self.settings_prev_button.bind(
            DGG.ENTER,
            self._hover_settings_row,
            [SettingsRow.CAMERA],
        )
        self.settings_next_button.bind(
            DGG.ENTER,
            self._hover_settings_row,
            [SettingsRow.CAMERA],
        )

        self.countdown_panel = DirectFrame(
            parent=self.aspect2d,
            frameSize=(-1.78, 1.78, -1.0, 1.0),
            frameColor=(0.01, 0.015, 0.04, 0.56),
        )
        self.countdown_text = OnscreenText(
            text="3",
            pos=(0.0, 0.08),
            scale=0.30,
            align=TextNode.ACenter,
            fg=(0.96, 0.97, 1.0, 1.0),
            shadow=(0.80, 0.03, 0.06, 0.95),
            **common,
        )
        self.countdown_status = OnscreenText(
            text="GET READY",
            pos=(0.0, -0.18),
            scale=0.050,
            align=TextNode.ACenter,
            fg=(0.62, 0.72, 0.96, 1.0),
            **common,
        )

        self.hud_text = OnscreenText(
            text="",
            pos=(-1.30, 0.91),
            scale=0.046,
            align=TextNode.ALeft,
            fg=(0.91, 0.94, 1.0, 1.0),
            shadow=(0.02, 0.02, 0.04, 0.9),
            **common,
        )
        self.death_text = OnscreenText(
            text="",
            pos=(0.0, 0.12),
            scale=0.075,
            align=TextNode.ACenter,
            fg=(1.0, 0.25, 0.24, 1.0),
            shadow=(0.02, 0.02, 0.04, 0.95),
            **common,
        )

        self._ui_nodes = [
            self.loading_panel,
            self.loading_title,
            self.loading_text,
            self.loading_bar,
            self.loading_hint,
            self.title_panel,
            self.title_text,
            self.title_subtitle,
            *self.title_buttons,
            self.title_help,
            self.title_nav_hint,
            self.asset_text,
            self.camera_border,
            self.camera_image,
            self.camera_header,
            self.camera_status,
            self.camera_help,
            self.tutorial_panel,
            self.tutorial_counter,
            self.tutorial_title,
            self.tutorial_hint,
            self.tutorial_status,
            self.tutorial_prompt,
            self.tutorial_progress,
            self.tutorial_escape,
            self.thwip_reference,
            self.settings_panel,
            self.settings_title,
            self.settings_subtitle,
            self.settings_source_heading,
            self.settings_source_frame,
            self.settings_source_value,
            self.settings_source_position,
            self.settings_prev_button,
            self.settings_next_button,
            self.settings_source_hint,
            self.settings_apply_button,
            self.settings_back_button,
            self.settings_status,
            self.settings_footer,
            self.countdown_panel,
            self.countdown_text,
            self.countdown_status,
            self.hud_text,
            self.death_text,
        ]
        self._refresh_title_menu()

    def _build_thwip_reference(self, centre_x: float) -> Any:
        """Load the supplied transparent web-shooter reference artwork."""

        if not DEFAULT_THWIP_IMAGE.is_file():
            raise GameStartupError(
                f"Tutorial reference image is missing: {DEFAULT_THWIP_IMAGE}"
            )
        try:
            surface = pygame.image.load(str(DEFAULT_THWIP_IMAGE))
            width, height = surface.get_size()
            # Panda stores four-channel texture RAM in BGRA byte order on the
            # supported desktop backends. Preserve the PNG alpha while keeping
            # Spider-Man's red glove red rather than swapping it to blue.
            bgra_bottom_up = pygame.image.tobytes(surface, "BGRA", True)
        except (OSError, pygame.error) as exc:
            raise GameStartupError(
                f"Could not load tutorial reference image: {exc}"
            ) from exc
        texture = Texture("web-shooter-reference")
        texture.setup2dTexture(
            width,
            height,
            Texture.T_unsigned_byte,
            Texture.F_rgba8,
        )
        texture.setRamImage(bgra_bottom_up)
        image = OnscreenImage(
            parent=self.aspect2d,
            image=texture,
            pos=(centre_x, 0.0, 0.10),
            scale=(0.235, 1.0, 0.235),
        )
        image.setTransparency(TransparencyAttrib.MAlpha)
        image.setDepthTest(False)
        image.setDepthWrite(False)
        return image

    def _refresh_title_menu(self) -> None:
        for index, button in enumerate(self.title_buttons):
            selected = index == self.title_menu.selected
            button["frameColor"] = (
                (0.72, 0.04, 0.08, 0.98)
                if selected
                else (0.05, 0.07, 0.14, 0.96)
            )
            button["text_fg"] = (
                (1.0, 1.0, 1.0, 1.0)
                if selected
                else (0.72, 0.79, 0.92, 1.0)
            )

    def _hover_title_button(self, index: int, *_event: Any) -> None:
        if self.state is GameState.TITLE:
            self.title_menu.selected = int(index) % len(self.title_menu.options)
            self._refresh_title_menu()

    def _activate_title_button(self, index: int) -> None:
        if self.state is not GameState.TITLE:
            return
        self.title_menu.selected = int(index) % len(self.title_menu.options)
        self._refresh_title_menu()
        self._activate_title_intent(self.title_menu.activate())

    def _bind_controls(self) -> None:
        self.accept("enter", self._on_enter)
        self.accept("escape", self._on_escape)
        self.accept("r", self._on_restart)
        self.accept("t", self._on_training)
        self.accept("arrow_up", self._on_menu_key, ["arrow_up"])
        self.accept("arrow_down", self._on_menu_key, ["arrow_down"])
        self.accept("w", self._on_menu_key, ["w"])
        self.accept("s", self._on_menu_key, ["s"])
        self.accept("arrow_left", self._on_settings_key, ["arrow_left"])
        self.accept("arrow_right", self._on_settings_key, ["arrow_right"])
        self.accept("a", self._on_settings_key, ["a"])
        self.accept("d", self._on_settings_key, ["d"])
        self.accept("tab", self._on_settings_key, ["tab"])
        self.accept("shift-tab", self._on_settings_key, ["shift_tab"])
        self.accept("[", self._on_camera_shortcut, [-1])
        self.accept("]", self._on_camera_shortcut, [1])
        latch_punch = getattr(self.producer, "latch_punch", None)
        if latch_punch is not None:
            self.accept("f", latch_punch)
            self.accept("mouse3", latch_punch)

    # --------------------------------------------------------------- states

    def _set_state(self, state: GameState, *, reset: bool = False) -> None:
        if reset:
            self._reset_run()
        previous = self.state
        self.state = state
        loading = state is GameState.LOADING
        title = state is GameState.TITLE
        tutorial = state is GameState.TUTORIAL
        settings = state is GameState.SETTINGS
        countdown = state is GameState.COUNTDOWN
        playing = state is GameState.PLAYING
        dead = state is GameState.DEAD

        # Theme carries the menus and fades once the countdown begins, so play
        # starts on the swing cues alone. Both calls are idempotent, and the
        # impact cue is on a channel stop() leaves alone so it survives the
        # cleanup that dying triggers.
        if title or tutorial or settings:
            self.audio.play_menu_music()
        else:
            self.audio.stop_music()
        if dead and previous is not GameState.DEAD:
            self.audio.play_fall()

        loading_nodes = (
            self.loading_panel,
            self.loading_title,
            self.loading_text,
            self.loading_bar,
            self.loading_hint,
        )
        title_nodes = (
            self.title_panel,
            self.title_text,
            self.title_subtitle,
            *self.title_buttons,
            self.title_help,
            self.title_nav_hint,
            self.asset_text,
        )
        camera_nodes = (
            self.camera_border,
            self.camera_image,
            self.camera_header,
            self.camera_status,
            self.camera_help,
        )
        tutorial_nodes = (
            self.tutorial_panel,
            self.tutorial_counter,
            self.tutorial_title,
            self.tutorial_hint,
            self.tutorial_status,
            self.tutorial_prompt,
            self.tutorial_progress,
            self.tutorial_escape,
        )
        settings_nodes = (
            self.settings_panel,
            self.settings_title,
            self.settings_subtitle,
            self.settings_source_heading,
            self.settings_source_frame,
            self.settings_source_value,
            self.settings_source_position,
            self.settings_prev_button,
            self.settings_next_button,
            self.settings_source_hint,
            self.settings_apply_button,
            self.settings_back_button,
            self.settings_status,
            self.settings_footer,
        )
        countdown_nodes = (
            self.countdown_panel,
            self.countdown_text,
            self.countdown_status,
        )
        for nodes, visible in (
            (loading_nodes, loading),
            (title_nodes, title),
            (
                camera_nodes,
                self.game_config.vision and (title or tutorial or settings),
            ),
            (tutorial_nodes, tutorial),
            (settings_nodes, settings),
            (countdown_nodes, countdown),
        ):
            for node in nodes:
                (node.show if visible else node.hide)()

        show_reference = tutorial and self.tutorial.step is TutorialStep.HOLD
        (
            self.thwip_reference.show
            if show_reference
            else self.thwip_reference.hide
        )()
        (self.hud_text.show if playing or dead else self.hud_text.hide)()
        (self.death_text.show if dead else self.death_text.hide)()

        if loading or title or tutorial or settings or countdown:
            self.audio.stop()
            self._remove_web()
            if self.character_controller is not None:
                self.character_controller.reset("idle")
            self._title_distance = max(self._title_distance, self.sim.z)
            self._sync_buildings(self.backdrop_world)
        if title:
            self._refresh_title_menu()
        elif tutorial:
            self._refresh_tutorial_ui()
        elif settings:
            self._refresh_settings_ui()
        elif countdown:
            self.countdown.reset()
            self.countdown_text.setText(self.countdown.text)
            self.countdown_status.setText(self.countdown.status)
        elif playing:
            self._sync_buildings(self.world)
            self._camera_ready = False
        elif dead:
            self.audio.stop()
            if self.character_controller is not None:
                self.character_controller.reset("fall")
            self.death_text.setText(
                f"{self.sim.death_reason.upper()}\n"
                f"{self.sim.z:.0f} m    BEST {self.best_distance:.0f} m\n\n"
                "R  RESTART    T  TRAINING    ENTER / ESC  TITLE"
            )

    def _reset_run(self) -> None:
        self.audio.stop()
        self.world = WorldStrip(seed=self.game_config.seed)
        self.sim = SwingSim()
        self.world.update(self.sim.z + 60.0)
        self._sync_buildings(self.world)
        self._remove_web()
        self._camera_ready = False
        if self.character_controller is not None:
            self.character_controller.reset("idle")

    def _on_enter(self) -> None:
        if self.state is GameState.TITLE:
            self._activate_title_intent(self.title_menu.activate())
        elif self.state is GameState.SETTINGS:
            self._handle_settings_intent(
                self._require_settings_menu().handle_key("enter")
            )
        elif self.state is GameState.DEAD:
            self._set_state(GameState.TITLE)

    def _on_escape(self) -> None:
        if self.state in {GameState.LOADING, GameState.TITLE}:
            self.request_quit()
        elif self.state is GameState.SETTINGS:
            menu = self._require_settings_menu()
            menu.discard()
            self.settings_message = ""
            self._set_state(GameState.TITLE)
        else:
            self._set_state(GameState.TITLE)

    def _on_restart(self) -> None:
        if self.state in {GameState.PLAYING, GameState.DEAD}:
            self._set_state(GameState.PLAYING, reset=True)

    def _on_training(self) -> None:
        if self.state is GameState.TITLE:
            self.title_menu.select(MenuIntent.TRAINING)
            self._refresh_title_menu()
            self._activate_title_intent(MenuIntent.TRAINING)
        elif self.state is GameState.DEAD:
            self.tutorial.reset()
            self._set_state(GameState.TUTORIAL)

    def _on_menu_key(self, key: str) -> None:
        if self.state is GameState.SETTINGS:
            self._on_settings_key(key)
            return
        if self.state is not GameState.TITLE:
            return
        intent = self.title_menu.handle_key(key)
        self._refresh_title_menu()
        if intent is not None:
            self._activate_title_intent(intent)

    def _activate_title_intent(self, intent: MenuIntent) -> None:
        if intent is MenuIntent.START:
            self._begin_countdown(reset=True)
        elif intent is MenuIntent.TRAINING:
            self.tutorial.reset()
            self._set_state(GameState.TUTORIAL)
        elif intent is MenuIntent.SETTINGS:
            self._require_settings_menu().discard()
            self.settings_message = ""
            self._set_state(GameState.SETTINGS)
        elif intent is MenuIntent.QUIT:
            self.request_quit()

    def _require_settings_menu(self) -> SettingsMenu:
        if self.settings_menu is None:
            raise GameStartupError("Settings were opened before input initialisation")
        return self.settings_menu

    def _on_settings_key(self, key: str) -> None:
        if self.state is not GameState.SETTINGS:
            return
        intent = self._require_settings_menu().handle_key(key)
        self._refresh_settings_ui()
        self._handle_settings_intent(intent)

    def _on_camera_shortcut(self, delta: int) -> None:
        if self.state is GameState.SETTINGS:
            self._cycle_settings_camera(delta)
        else:
            self._switch_camera(delta)

    def _hover_settings_row(
        self,
        row: SettingsRow,
        *_event: Any,
    ) -> None:
        if self.state is GameState.SETTINGS:
            self._require_settings_menu().select_row(row)
            self._refresh_settings_ui()

    def _cycle_settings_camera(self, delta: int) -> None:
        if self.state is not GameState.SETTINGS:
            return
        menu = self._require_settings_menu()
        menu.select_row(SettingsRow.CAMERA)
        menu.cycle_camera(delta)
        self.settings_message = (
            "Selection changed. Choose APPLY CAMERA to test this source."
            if menu.dirty
            else "This is the active camera source."
        )
        self._refresh_settings_ui()

    def _activate_settings_apply(self) -> None:
        if self.state is not GameState.SETTINGS:
            return
        self._require_settings_menu().select_row(SettingsRow.APPLY)
        self._refresh_settings_ui()
        self._handle_settings_intent(SettingsIntent.APPLY)

    def _activate_settings_back(self) -> None:
        if self.state is not GameState.SETTINGS:
            return
        self._require_settings_menu().select_row(SettingsRow.BACK)
        self._refresh_settings_ui()
        self._handle_settings_intent(SettingsIntent.BACK)

    def _handle_settings_intent(
        self,
        intent: SettingsIntent | None,
    ) -> None:
        if intent is None:
            return
        menu = self._require_settings_menu()
        if intent is SettingsIntent.BACK:
            menu.discard()
            self.settings_message = ""
            self._set_state(GameState.TITLE)
            return

        target = menu.pending_camera
        if not self.game_config.vision or target is None:
            self.settings_message = (
                "Camera input is disabled. Restart without --keyboard to use "
                "MediaPipe gestures."
            )
            self._refresh_settings_ui()
            return
        if target == self.camera_index:
            menu.commit()
            self.settings_message = f"Camera {target} is already active."
            self._refresh_settings_ui()
            return

        self.settings_message = f"Testing camera {target}..."
        self._refresh_settings_ui()
        try:
            self.graphicsEngine.renderFrame()
        except Exception:
            pass
        if self._switch_camera_to(target):
            menu.commit()
            self.settings_message = f"Camera {target} is now active."
        else:
            self.settings_message = (
                self.camera_notice or f"Camera {target} could not be opened."
            )
        self._refresh_settings_ui()

    def _begin_countdown(self, *, reset: bool) -> None:
        if reset:
            self._reset_run()
        self._set_state(GameState.COUNTDOWN)

    # --------------------------------------------------------------- frame

    def _update(self, task: Any) -> Any:
        if self._closed:
            return task.done

        dt = min(max(float(self.clock.getDt()), 0.0), MAX_FRAME_DT)
        self._poll_control()
        self._update_vision_health()

        if self.state is GameState.LOADING:
            self._update_loading(dt)
        elif self.state is GameState.TITLE:
            self._update_title(dt)
        elif self.state is GameState.TUTORIAL:
            self._update_tutorial(dt)
        elif self.state is GameState.SETTINGS:
            self._update_settings(dt)
        elif self.state is GameState.COUNTDOWN:
            self._update_countdown(dt)
        elif self.state is GameState.PLAYING:
            self._update_playing(dt, self.control)
        else:
            self._update_player_node()
            self._update_camera(dt)
            self._update_hud()

        self._frames += 1
        if (
            self.game_config.max_frames is not None
            and self._frames >= self.game_config.max_frames
        ):
            self.request_quit()
            return task.done
        return task.cont

    def _poll_control(self) -> None:
        if self.producer is None:
            return
        try:
            self.control = self.producer.poll()
        except Exception as exc:
            if self.game_config.vision:
                self.vision_error = f"camera input failed: {exc}"

    def _update_vision_health(self) -> None:
        if not self.game_config.vision or self.producer is None:
            return
        error = getattr(self.producer, "error", None)
        if error:
            self.vision_error = str(error)
            self._vision_ready = False
            return
        try:
            ready = bool(self.producer.wait_ready(0.0))
        except Exception as exc:
            self.vision_error = f"camera startup failed: {exc}"
            self._vision_ready = False
            return
        if ready:
            newly_ready = not self._vision_ready
            self._vision_ready = True
            if newly_ready:
                self.vision_error = None

    def _update_loading(self, dt: float) -> None:
        self._loading_elapsed += dt
        self._update_menu_backdrop(dt)

        terminal = self._vision_ready
        if self.game_config.vision:
            error = getattr(self.producer, "error", None)
            timed_out = time.monotonic() >= self._vision_deadline
            if error:
                self.vision_error = str(error)
            elif timed_out and not self._vision_ready:
                self.vision_error = (
                    f"camera {self.camera_index} timed out; open SETTINGS on "
                    "the title screen to try another source"
                )
            terminal = self._vision_ready or bool(self.vision_error)
            message = (
                f"CAMERA {self.camera_index} READY"
                if self._vision_ready
                else (
                    "CAMERA NEEDS ATTENTION"
                    if self.vision_error
                    else f"CONNECTING TO CAMERA {self.camera_index}"
                )
            )
            self.loading_text.setText(message)
        else:
            sound = "SOUND READY" if self.audio.enabled else "SILENT MODE"
            self.loading_text.setText(f"CITY, HERO AND CONTROLS READY  //  {sound}")

        settling = min(1.0, self._loading_elapsed / LOADING_MIN_SECONDS)
        self.loading_bar["value"] = 88.0 + settling * 12.0
        if terminal and self._loading_elapsed >= LOADING_MIN_SECONDS:
            if self.game_config.skip_title and (
                not self.game_config.vision or self._vision_ready
            ):
                self._begin_countdown(reset=True)
            else:
                self._set_state(GameState.TITLE)

    def _update_title(self, dt: float) -> None:
        self._update_menu_backdrop(dt)
        self._update_camera_preview()

    def _update_tutorial(self, dt: float) -> None:
        self._update_menu_backdrop(dt)
        self._update_camera_preview()
        if self.tutorial.update(self.control, dt):
            self._begin_countdown(reset=True)
            return
        self._refresh_tutorial_ui()

    def _update_settings(self, dt: float) -> None:
        self._update_menu_backdrop(dt)
        self._update_camera_preview()
        self._refresh_settings_ui()

    def _update_countdown(self, dt: float) -> None:
        self._update_menu_backdrop(dt)
        if self.countdown.update(dt):
            self._set_state(GameState.PLAYING)
            self._update_player_node()
            self._update_camera(0.0)
            self._update_hud()
            self.road.setY(self.sim.z)
            return
        self.countdown_text.setText(self.countdown.text)
        self.countdown_status.setText(self.countdown.status)

    def _update_menu_backdrop(self, dt: float) -> None:
        self._title_distance += 32.0 * dt
        self.backdrop_world.update(self._title_distance + 60.0)
        self._sync_buildings(self.backdrop_world)
        lateral = math.sin(self._title_distance * 0.018) * 2.4
        self.player.setPos(
            *simulation_to_render(-5.5, T.START_Y, self._title_distance + 8.0)
        )
        self.player.setHpr(0.0, 0.0, 0.0)
        self.road.setY(self._title_distance)

        desired = Vec3(7.5 + lateral, self._title_distance - 18.0, T.START_Y + 7.5)
        focus = Vec3(0.0, self._title_distance + 7.0, T.START_Y + 0.4)
        self.camera.setPos(desired)
        self.camera.lookAt(focus)
        self.camLens.setFov(76.0)
        self._camera_ready = False

    def _refresh_tutorial_ui(self) -> None:
        view = self.tutorial.view
        self.tutorial_counter.setText(
            f"TUTORIAL {view.step_number} / {view.step_count}"
        )
        self.tutorial_title.setText(view.title)
        self.tutorial_hint.setText(view.hint)
        self.tutorial_status.setText(view.status)
        self.tutorial_prompt.setText(view.prompt)
        self.tutorial_progress["value"] = view.progress * 100.0

        if "NO HAND" in view.status or "KEEP YOUR HAND" in view.status:
            colour = (1.0, 0.25, 0.24, 1.0)
        elif any(
            marker in view.status
            for marker in ("DETECTED", "COMPLETE", "TRACKING", "RANGE")
        ):
            colour = (0.18, 0.92, 0.56, 1.0)
        else:
            colour = (1.0, 0.72, 0.20, 1.0)
        self.tutorial_status.setFg(colour)

        show_reference = (
            self.state is GameState.TUTORIAL
            and view.step is TutorialStep.HOLD
        )
        (
            self.thwip_reference.show
            if show_reference
            else self.thwip_reference.hide
        )()

    def _refresh_settings_ui(self) -> None:
        menu = self._require_settings_menu()
        view = menu.view
        self.settings_source_heading.setText(view.camera_heading)
        self.settings_source_value.setText(view.camera_value)
        self.settings_source_position.setText(view.camera_position)
        self.settings_source_hint.setText(view.camera_hint)

        camera_focused = view.focused_row is SettingsRow.CAMERA
        apply_focused = view.focused_row is SettingsRow.APPLY
        back_focused = view.focused_row is SettingsRow.BACK
        self.settings_source_frame["frameColor"] = (
            (0.30, 0.045, 0.075, 0.98)
            if camera_focused
            else (0.045, 0.060, 0.125, 0.98)
        )
        for button in (self.settings_prev_button, self.settings_next_button):
            button["frameColor"] = (
                (0.72, 0.04, 0.08, 0.98)
                if camera_focused
                else (0.08, 0.11, 0.22, 0.98)
            )
            button["state"] = (
                DGG.NORMAL if view.can_change_camera else DGG.DISABLED
            )
        self.settings_apply_button["frameColor"] = (
            (0.72, 0.04, 0.08, 0.98)
            if apply_focused
            else (
                (0.38, 0.045, 0.075, 0.98)
                if view.dirty
                else (0.05, 0.07, 0.14, 0.98)
            )
        )
        self.settings_back_button["frameColor"] = (
            (0.72, 0.04, 0.08, 0.98)
            if back_focused
            else (0.05, 0.07, 0.14, 0.98)
        )

        if self.settings_message:
            message = _short_camera_message(self.settings_message, limit=92)
        elif self.game_config.vision:
            message = (
                f"ACTIVE: CAMERA {self.camera_index}  //  "
                "Preview changes only after Apply."
            )
        else:
            message = (
                "CAMERA INPUT DISABLED  //  Launch without --keyboard to "
                "enable MediaPipe."
            )
        self.settings_status.setText(message)
        if any(word in message.lower() for word in ("unavailable", "failed", "disabled")):
            colour = (1.0, 0.30, 0.28, 1.0)
        elif "now active" in message.lower() or "already active" in message.lower():
            colour = (0.18, 0.92, 0.56, 1.0)
        else:
            colour = (0.58, 0.70, 0.92, 1.0)
        self.settings_status.setFg(colour)

    def _update_camera_preview(self) -> None:
        if not self.game_config.vision or self.producer is None:
            return
        if (
            self.camera_notice
            and time.monotonic() >= self._camera_notice_until
        ):
            self.camera_notice = None

        frame = None
        worker = getattr(self.producer, "worker", None)
        debug_frame = getattr(worker, "debug_frame", None)
        if callable(debug_frame):
            try:
                frame, _landmarks = debug_frame()
            except Exception as exc:
                self.vision_error = f"camera preview failed: {exc}"

        if frame is not None and id(frame) != self._camera_frame_token:
            try:
                height, width = frame.shape[:2]
                if len(frame.shape) != 3 or frame.shape[2] < 3:
                    raise ValueError("expected a three-channel BGR frame")
                # VisionWorker has already mirrored the frame. Panda textures
                # use a bottom-left origin, so reverse rows while changing BGR
                # to RGB for an upright, natural preview.
                rgb_bottom_up = frame[::-1, :, 2::-1].tobytes()
                texture_size = (int(width), int(height))
                if texture_size != self._camera_texture_size:
                    self._camera_texture.setup2dTexture(
                        *texture_size,
                        Texture.T_unsigned_byte,
                        Texture.F_rgb8,
                    )
                    self._camera_texture_size = texture_size
                    source_aspect = width / max(1.0, float(height))
                    box_aspect = 0.54 / 0.35
                    if source_aspect >= box_aspect:
                        image_x = 0.54
                        image_z = image_x / source_aspect
                    else:
                        image_z = 0.35
                        image_x = image_z * source_aspect
                    self.camera_image.setScale(image_x, 1.0, image_z)
                self._camera_texture.setRamImage(rgb_bottom_up)
                self._camera_frame_token = id(frame)
                self._camera_has_frame = True
                if self.vision_error and self.vision_error.startswith(
                    "camera preview failed"
                ):
                    self.vision_error = None
            except Exception as exc:
                self.vision_error = f"camera preview failed: {exc}"

        auto = "  //  AUTO" if self.auto_picked_camera else ""
        self.camera_header.setText(f"LIVE CAMERA {self.camera_index}{auto}")
        if self.vision_error:
            status = "CAMERA NEEDS ATTENTION"
            colour = (1.0, 0.25, 0.24, 1.0)
            help_text = _short_camera_message(
                f"{self.vision_error}  |  Open SETTINGS to choose another source"
            )
        elif not self._camera_has_frame:
            status = "CAMERA STARTING"
            colour = (1.0, 0.72, 0.20, 1.0)
            help_text = "Keep your full hand in frame"
        elif self.control.tracking_lost:
            status = "NO HAND - MOVE INTO FRAME"
            colour = (1.0, 0.25, 0.24, 1.0)
            help_text = "Show your full palm and fingertips to the camera"
        elif self.control.thwip_held:
            status = "WEB-SHOOTER DETECTED"
            colour = (0.18, 0.92, 0.56, 1.0)
            help_text = "Gesture locked - release to detach the web"
        else:
            status = "HAND FOUND - MAKE THE SIGN"
            colour = (1.0, 0.72, 0.20, 1.0)
            help_text = "Index + pinky out; curl middle + ring"
        if self.camera_notice and not self.vision_error:
            help_text = _short_camera_message(self.camera_notice)
        if not self.vision_error and "SETTINGS" not in help_text.upper():
            help_text = _short_camera_message(
                f"{help_text}  |  Open SETTINGS to change camera"
            )
        self.camera_status.setText(status)
        self.camera_status.setFg(colour)
        self.camera_help.setText(help_text)

    def _switch_camera(self, delta: int) -> None:
        if not _camera_switch_allowed(self.state, self.game_config.vision):
            return
        current_index = 0 if self.camera_index is None else self.camera_index
        new_index = max(0, current_index + int(delta))
        if new_index == current_index:
            return
        self._switch_camera_to(new_index)

    def _switch_camera_to(self, new_index: int) -> bool:
        if not _camera_switch_allowed(self.state, self.game_config.vision):
            return False
        current_index = 0 if self.camera_index is None else self.camera_index
        new_index = max(0, int(new_index))
        if new_index == current_index:
            return True

        self.camera_status.setText(f"CONNECTING TO CAMERA {new_index}")
        self.camera_status.setFg((1.0, 0.72, 0.20, 1.0))
        self.camera_help.setText("Testing the camera before switching...")
        try:
            self.graphicsEngine.renderFrame()
        except Exception:
            pass

        from spidergame.producers.vision import VisionProducer

        candidate = None
        try:
            candidate = VisionProducer(camera_index=new_index, keep_frame=True)
            if not candidate.wait_ready(VISION_SWITCH_TIMEOUT) or candidate.error:
                raise RuntimeError(candidate.error or "timed out")
        except Exception as exc:
            if candidate is not None:
                candidate.close()
            self.camera_notice = _short_camera_message(
                f"camera {new_index} unavailable ({exc}); "
                f"still using camera {current_index}"
            )
            self._camera_notice_until = time.monotonic() + 6.0
            self._update_camera_preview()
            return False

        old = self.producer
        self.producer = candidate
        self.camera_index = new_index
        self.auto_picked_camera = False
        self.vision_error = None
        self.camera_notice = f"Camera {new_index} ready."
        self._camera_notice_until = time.monotonic() + 4.0
        self._vision_ready = True
        self._camera_has_frame = False
        self._camera_frame_token = None
        self._camera_texture_size = None
        self.control = ControlState(tracking_lost=True)
        if old is not None:
            old.close()
        if self.state is not GameState.SETTINGS:
            self.settings_menu = self._create_settings_menu()
        self._update_camera_preview()
        return True

    def _update_playing(self, dt: float, control: ControlState) -> None:
        self.world.update(self.sim.z + 60.0)
        events = self.sim.update(dt, control, self.world)
        self.audio.handle(events, self.sim)
        if self.character_controller is not None:
            self.character_controller.update(self.sim, events, dt)
        self.best_distance = max(self.best_distance, self.sim.z)
        self._sync_buildings(self.world)
        self._update_player_node()
        self._update_web()
        self._update_camera(dt)
        self._update_hud()
        self.road.setY(self.sim.z)

        if not self.sim.alive:
            self._set_state(GameState.DEAD)

    def _sync_buildings(self, world: WorldStrip | None = None) -> None:
        try:
            self.building_renderer.sync(self.world if world is None else world)
        except Exception as exc:
            raise GameStartupError(f"Building renderer sync failed: {exc}") from exc

    def _update_player_node(self) -> None:
        self.player.setPos(*simulation_to_render(self.sim.x, self.sim.y, self.sim.z))
        self.player.setHpr(
            *velocity_to_render_hpr(self.sim.vx, self.sim.vy, self.sim.vz)
        )

    def _update_web(self) -> None:
        self._remove_web()
        anchor = self.sim.anchor
        if anchor is None:
            return

        line = LineSegs("web")
        line.setThickness(2.5)
        line.setColor(0.92, 0.96, 1.0, 1.0)
        player = simulation_to_render(self.sim.x, self.sim.y + 1.55, self.sim.z)
        if self.character_controller is not None:
            try:
                player = self.character_controller.hand_world_position(anchor)
            except Exception:
                # Presentation should never interrupt the simulation.  The
                # torso-height fallback remains visually usable if an exported
                # hand socket is absent or temporarily invalid.
                pass
        target = simulation_to_render(anchor.x, anchor.y, anchor.z)
        line.moveTo(*player)
        line.drawTo(*target)
        self._web_node = self.render.attachNewNode(line.create())
        self._web_node.setLightOff()

    def _remove_web(self) -> None:
        if self._web_node is not None:
            self._web_node.removeNode()
            self._web_node = None

    def _update_camera(self, dt: float) -> None:
        player = Vec3(*simulation_to_render(self.sim.x, self.sim.y, self.sim.z))
        desired = player + Vec3(0.0, -CAMERA_BACK, CAMERA_UP)
        focus = player + Vec3(0.0, CAMERA_LOOK_AHEAD, 1.0)

        if not self._camera_ready:
            self.camera.setPos(desired)
            self._camera_ready = True
        else:
            blend = 1.0 - math.exp(-CAMERA_RESPONSE * dt)
            self.camera.setPos(self.camera.getPos() + (desired - self.camera.getPos()) * blend)
        self.camera.lookAt(focus)
        self.camera.setR(max(-5.5, min(5.5, -self.sim.vx * 0.075)))

        speed_ratio = min(
            max((self.sim.vz - T.START_SPEED) / (T.MAX_SPEED - T.START_SPEED), 0.0),
            1.0,
        )
        self.camLens.setFov(76.0 + 10.0 * speed_ratio)

    def _update_hud(self) -> None:
        web = "WEB ATTACHED" if self.sim.attached else "FALLING"
        controls = (
            "web-shooter gesture  web    R  restart    ESC  title"
            if self.game_config.vision
            else "SPACE / mouse 1  web    R  restart    ESC  title"
        )
        self.hud_text.setText(
            f"{self.sim.z:7.0f} m\n"
            f"altitude  {self.sim.y:5.1f}\n"
            f"{web}\n"
            f"{controls}"
        )

    # --------------------------------------------------------------- cleanup

    def request_quit(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.taskMgr.stop()

    def close(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        self._shutdown_complete = True
        try:
            producer = getattr(self, "producer", None)
            if producer is not None:
                producer.close()
        finally:
            try:
                if hasattr(self, "audio"):
                    self.audio.close()
            finally:
                try:
                    if pygame.mixer.get_init() is not None:
                        pygame.mixer.quit()
                except pygame.error:
                    pass
                self._remove_web()
                if self.character_controller is not None:
                    self.character_controller.cleanup()
                try:
                    super().destroy()
                except Exception:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spider Swing - animated third-person Panda3D game"
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--building-asset",
        default=str(DEFAULT_BUILDING_ASSET),
        help="city GLB (default: assets/models/buildings/new_york_generic.glb)",
    )
    parser.add_argument(
        "--building-manifest",
        default=str(DEFAULT_BUILDING_MANIFEST),
        help="JSON node manifest used by BuildingRenderer",
    )
    parser.add_argument(
        "--character",
        default=str(DEFAULT_CHARACTER_ASSET),
        help="optional character GLB; a tall placeholder is used if missing",
    )
    parser.add_argument(
        "--character-manifest",
        default=None,
        help=(
            "JSON animation, pivot and hand-socket manifest; the bundled model "
            "uses its default and custom models auto-discover an adjacent manifest"
        ),
    )
    parser.add_argument("--no-character", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--vision",
        action="store_true",
        dest="vision",
        help="use MediaPipe camera gestures (the normal windowed default)",
    )
    input_group.add_argument(
        "--keyboard",
        action="store_false",
        dest="vision",
        help="disable the camera and use keyboard + mouse controls",
    )
    parser.set_defaults(vision=None)
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="OpenCV camera index (default: auto-detect)",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="probe capture indices and print the available cameras",
    )
    parser.add_argument(
        "--skip-title",
        "--skip-tutorial",
        dest="skip_title",
        action="store_true",
        help="skip the front end and start at the 3-2-1 countdown",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="open an offscreen buffer (intended for smoke tests)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> GameConfig:
    if args.frames is not None and args.frames <= 0:
        raise GameStartupError("--frames must be a positive integer")
    if args.camera is not None and args.camera < 0:
        raise GameStartupError("--camera must be zero or greater")
    if args.vision is False and args.camera is not None:
        raise GameStartupError("--camera cannot be used together with --keyboard")
    character_asset = None if args.no_character else _resolved_path(args.character)
    character_manifest: Path | None = None
    if character_asset is not None:
        if args.character_manifest is not None:
            character_manifest = _resolved_path(args.character_manifest)

    # MediaPipe is the primary input path for the actual game. Headless smoke
    # tests remain camera-free unless --vision is explicitly requested.
    vision = (not args.headless) if args.vision is None else args.vision

    return GameConfig(
        seed=args.seed,
        building_asset=_resolved_path(args.building_asset),
        building_manifest=_resolved_path(args.building_manifest),
        character_asset=character_asset,
        character_manifest=character_manifest,
        audio=not args.no_audio,
        vision=vision,
        camera_index=args.camera,
        skip_title=args.skip_title,
        headless=args.headless,
        max_frames=args.frames,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise dependency and asset diagnostics."""

    try:
        args = build_parser().parse_args(argv)
        if args.list_cameras:
            from spidergame.vision import devices

            print("probing capture indices 0-3...")
            for info in devices.available():
                flag = "   <-- BLACK FRAMES" if info.dark else ""
                print(
                    f"  {info.label()}  brightness "
                    f"{info.brightness:5.1f}{flag}"
                )
            print("\nWindows device list (order does not match capture indices):")
            for name in devices.system_names():
                print(f"  {name}")
            return 0
        config = config_from_args(args)
        require_panda3d()
        validate_building_assets(config.building_asset, config.building_manifest)
        app = SpiderGame3D(config)
    except GameStartupError as exc:
        print(f"3D runner could not start: {exc}", file=sys.stderr)
        return 2

    try:
        app.run()
    except GameStartupError as exc:
        print(f"3D runner stopped: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        app.close()
    return 0


__all__ = [
    "GameConfig",
    "GameStartupError",
    "GameState",
    "PandaKeyboardProducer",
    "SpiderGame3D",
    "build_parser",
    "config_from_args",
    "main",
    "simulation_to_render",
    "validate_building_assets",
    "velocity_to_render_hpr",
]
