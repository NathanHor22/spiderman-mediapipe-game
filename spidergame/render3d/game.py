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


_PANDA_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
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

PLAYER_HEIGHT = 4.8
CAMERA_BACK = 18.0
CAMERA_UP = 6.2
CAMERA_LOOK_AHEAD = 7.0
CAMERA_RESPONSE = 7.5
ROAD_LENGTH = 1_200.0
MAX_FRAME_DT = 0.05


class GameStartupError(RuntimeError):
    """A dependency, window, or required building asset cannot be prepared."""


class GameState(Enum):
    """Small state machine shared by keyboard callbacks and the frame task."""

    TITLE = auto()
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
    vision: bool = False
    camera_index: int = 0
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
            "Panda3D is required for run_game_3d.py. Install it with "
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

    def handle(self, _events: Any, _sim: Any = None) -> None:
        pass

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
        self.best_distance = 0.0

        self._setup_lighting_and_fog()
        self._setup_road()
        self.player = self.render.attachNewNode("player-root")
        self._setup_character_lighting()
        self.character_controller = None
        self.character_note = self._load_character(
            config.character_asset,
            config.character_manifest,
        )
        self._web_node = None

        self.building_renderer = self._create_building_renderer()
        self.producer = self._create_producer()
        self.audio = self._create_audio()
        self._setup_ui()
        self._bind_controls()

        self.world = WorldStrip(seed=config.seed)
        self.sim = SwingSim()
        self.world.update(self.sim.z + 60.0)
        self._sync_buildings()

        self.state = GameState.TITLE
        self._set_state(
            GameState.PLAYING if config.skip_title else GameState.TITLE,
            reset=config.skip_title,
        )
        self.taskMgr.add(self._update, "spidergame-3d-update", sort=10)

    # --------------------------------------------------------------- startup

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
            return PandaKeyboardProducer(self)

        from spidergame.producers.vision import VisionProducer

        producer = VisionProducer(
            camera_index=self.game_config.camera_index,
            keep_frame=False,
        )
        if producer.wait_ready(timeout=15.0):
            return producer

        error = producer.error or "camera did not deliver usable frames"
        producer.close()
        raise GameStartupError(
            f"could not start vision camera {self.game_config.camera_index}: "
            f"{error}. Choose another device with --camera N or run without "
            "--vision for keyboard controls."
        )

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
        self.title_text = OnscreenText(
            text="SPIDER\nENDLESS SWINGER",
            pos=(0.0, 0.34),
            scale=0.13,
            align=TextNode.ACenter,
            fg=(0.95, 0.96, 1.0, 1.0),
            shadow=(0.02, 0.02, 0.04, 0.9),
            **common,
        )
        control_help = (
            "Show the web-shooter sign to fire\n"
            "Move your hand to aim  |  relax it to launch"
            if self.game_config.vision
            else (
                "Hold SPACE / mouse 1 to shoot a web\n"
                "Aim with the mouse  |  release to launch"
            )
        )
        self.title_help = OnscreenText(
            text=f"ENTER  START\n{control_help}\nESC  QUIT",
            pos=(0.0, -0.20),
            scale=0.052,
            align=TextNode.ACenter,
            fg=(0.78, 0.84, 0.95, 1.0),
            shadow=(0.02, 0.02, 0.04, 0.9),
            **common,
        )
        self.asset_text = OnscreenText(
            text=self.character_note,
            pos=(-1.30, -0.91),
            scale=0.035,
            align=TextNode.ALeft,
            fg=(0.58, 0.64, 0.74, 1.0),
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

    def _bind_controls(self) -> None:
        self.accept("enter", self._on_enter)
        self.accept("escape", self._on_escape)
        self.accept("r", self._on_restart)
        latch_punch = getattr(self.producer, "latch_punch", None)
        if latch_punch is not None:
            self.accept("f", latch_punch)
            self.accept("mouse3", latch_punch)

    # --------------------------------------------------------------- states

    def _set_state(self, state: GameState, *, reset: bool = False) -> None:
        if reset:
            self._reset_run()
        self.state = state
        title = state is GameState.TITLE
        playing = state is GameState.PLAYING
        dead = state is GameState.DEAD

        (self.title_text.show if title else self.title_text.hide)()
        (self.title_help.show if title else self.title_help.hide)()
        (self.asset_text.show if title else self.asset_text.hide)()
        (self.hud_text.show if playing or dead else self.hud_text.hide)()
        (self.death_text.show if dead else self.death_text.hide)()

        if title:
            self.audio.stop()
            self._remove_web()
            if self.character_controller is not None:
                self.character_controller.reset("idle")
            self._title_distance = max(self._title_distance, self.sim.z)
        elif dead:
            self.audio.stop()
            if self.character_controller is not None:
                self.character_controller.reset("fall")
            self.death_text.setText(
                f"{self.sim.death_reason.upper()}\n"
                f"{self.sim.z:.0f} m    BEST {self.best_distance:.0f} m\n\n"
                "R  RESTART    ENTER / ESC  TITLE"
            )

    def _reset_run(self) -> None:
        self.audio.stop()
        self.world = WorldStrip(seed=self.game_config.seed)
        self.sim = SwingSim()
        self.world.update(self.sim.z + 60.0)
        self._sync_buildings()
        self._remove_web()
        self._camera_ready = False
        if self.character_controller is not None:
            self.character_controller.reset("idle")

    def _on_enter(self) -> None:
        if self.state is GameState.TITLE:
            self._set_state(GameState.PLAYING, reset=True)
        elif self.state is GameState.DEAD:
            self._set_state(GameState.TITLE)

    def _on_escape(self) -> None:
        if self.state is GameState.TITLE:
            self.request_quit()
        else:
            self._set_state(GameState.TITLE)

    def _on_restart(self) -> None:
        if self.state in {GameState.PLAYING, GameState.DEAD}:
            self._set_state(GameState.PLAYING, reset=True)

    # --------------------------------------------------------------- frame

    def _update(self, task: Any) -> Any:
        if self._closed:
            return task.done

        dt = min(max(float(self.clock.getDt()), 0.0), MAX_FRAME_DT)
        if self.state is GameState.TITLE:
            self._update_title(dt)
        elif self.state is GameState.PLAYING:
            self._update_playing(dt)
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

    def _update_title(self, dt: float) -> None:
        self._title_distance += 20.0 * dt
        self.world.update(self._title_distance + 60.0)
        self._sync_buildings()
        # Keep the hero clear of the centered title/instructions while still
        # showing the third-person silhouette against the moving city.
        self.player.setPos(
            *simulation_to_render(-5.5, T.START_Y, self._title_distance)
        )
        self.player.setHpr(0.0, 0.0, 0.0)
        self.road.setY(self._title_distance)

        desired = Vec3(7.5, self._title_distance - 18.0, T.START_Y + 7.5)
        focus = Vec3(0.0, self._title_distance + 6.0, T.START_Y + 0.4)
        self.camera.setPos(desired)
        self.camera.lookAt(focus)
        self._camera_ready = False

    def _update_playing(self, dt: float) -> None:
        control = self.producer.poll()
        self.world.update(self.sim.z + 60.0)
        events = self.sim.update(dt, control, self.world)
        self.audio.handle(events, self.sim)
        if self.character_controller is not None:
            self.character_controller.update(self.sim, events, dt)
        self.best_distance = max(self.best_distance, self.sim.z)
        self._sync_buildings()
        self._update_player_node()
        self._update_web()
        self._update_camera(dt)
        self._update_hud()
        self.road.setY(self.sim.z)

        if not self.sim.alive:
            self._set_state(GameState.DEAD)

    def _sync_buildings(self) -> None:
        try:
            self.building_renderer.sync(self.world)
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
            if hasattr(self, "producer"):
                self.producer.close()
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
        description="Panda3D prototype for the Spider endless swinger"
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
    parser.add_argument(
        "--vision",
        action="store_true",
        help="use MediaPipe hand gestures instead of keyboard controls",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index used with --vision (default: 0)",
    )
    parser.add_argument("--skip-title", action="store_true")
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
    if args.camera < 0:
        raise GameStartupError("--camera must be zero or greater")
    character_asset = None if args.no_character else _resolved_path(args.character)
    character_manifest: Path | None = None
    if character_asset is not None:
        if args.character_manifest is not None:
            character_manifest = _resolved_path(args.character_manifest)

    return GameConfig(
        seed=args.seed,
        building_asset=_resolved_path(args.building_asset),
        building_manifest=_resolved_path(args.building_manifest),
        character_asset=character_asset,
        character_manifest=character_manifest,
        audio=not args.no_audio,
        vision=args.vision,
        camera_index=args.camera,
        skip_title=args.skip_title,
        headless=args.headless,
        max_frames=args.frames,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with concise dependency and asset diagnostics."""

    try:
        args = build_parser().parse_args(argv)
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
