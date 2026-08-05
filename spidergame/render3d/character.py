"""Rigged Panda3D character presentation for :class:`SwingSim`.

The simulation owns the character's world transform.  ``CharacterController``
parents a uniformly scaled, physics-pivot-aligned Actor beneath the NodePath
supplied by the caller and only changes animation state.  Embedded clips must be
in-place so animation can never feed root motion back into swing physics.

Panda3D and panda3d-gltf are imported lazily.  State, phase, clip, and joint
selection helpers therefore remain testable in the base Pygame environment.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUIRED_CLIPS = ("idle", "fall", "shoot", "attach", "swing", "release")
DEFAULT_SWING_ARC_RADIANS = math.radians(120.0)

ONE_SHOT_SECONDS = {
    "shoot": 0.24,
    "attach": 0.30,
    "release": 0.28,
}

_HAND_ALIASES = {
    "right": (
        "hand.R",
        "DEF-hand.R",
        "RightHand",
        "mixamorig:RightHand",
        "hand_r",
        "r_hand",
    ),
    "left": (
        "hand.L",
        "DEF-hand.L",
        "LeftHand",
        "mixamorig:LeftHand",
        "hand_l",
        "l_hand",
    ),
}


class CharacterControllerError(RuntimeError):
    """Base class for actionable character presentation failures."""


class CharacterAssetError(CharacterControllerError):
    """The model or manifest is missing, malformed, or incompatible."""


class Panda3DUnavailableError(CharacterControllerError):
    """The optional Panda3D character runtime is unavailable."""


class CharacterAnimState(str, Enum):
    IDLE = "idle"
    FALL = "fall"
    SHOOT = "shoot"
    ATTACH = "attach"
    SWING = "swing"
    RELEASE = "release"


@dataclass(frozen=True)
class CharacterBounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = self.minimum + self.maximum
        if not all(math.isfinite(value) for value in values):
            raise ValueError("character bounds must contain finite numbers")
        if any(upper <= lower for lower, upper in zip(self.minimum, self.maximum)):
            raise ValueError(
                "character bounds must have max greater than min on every axis"
            )

    @property
    def height(self) -> float:
        return self.maximum[2] - self.minimum[2]


@dataclass
class CharacterManifest:
    bounds: CharacterBounds | None = None
    physics_pivot: tuple[float, float, float] | None = None
    heading_offset_degrees: float = 0.0
    clips: dict[str, str] = field(default_factory=dict)
    joints: dict[str, str] = field(default_factory=dict)


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _vec3(value: Any, label: str) -> tuple[float, float, float]:
    if isinstance(value, Mapping):
        try:
            value = [value[axis] for axis in ("x", "y", "z")]
        except KeyError as exc:
            raise ValueError(f"{label} must contain x, y, and z") from exc
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a three-number sequence")
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three numbers")
    return tuple(
        _number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _bounds(value: Any, label: str) -> CharacterBounds:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")

    if "bounds" in value:
        return _bounds(value["bounds"], f"{label}.bounds")

    y_up = value.get("local_bounds_y_up", value.get("bounds_y_up"))
    if y_up is not None:
        source = _bounds(y_up, f"{label}.local_bounds_y_up")
        return CharacterBounds(
            (
                source.minimum[0],
                -source.maximum[2],
                source.minimum[1],
            ),
            (
                source.maximum[0],
                -source.minimum[2],
                source.maximum[1],
            ),
        )

    lower = value.get("min", value.get("minimum"))
    upper = value.get("max", value.get("maximum"))
    if lower is None or upper is None:
        raise ValueError(f"{label} must contain min and max vectors")
    return CharacterBounds(
        _vec3(lower, f"{label}.min"),
        _vec3(upper, f"{label}.max"),
    )


def _parse_clip_manifest(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    result: dict[str, str] = {}
    if isinstance(value, Mapping):
        for canonical, payload in value.items():
            key = str(canonical).strip().lower()
            if isinstance(payload, str):
                actual = payload
            elif isinstance(payload, Mapping):
                actual = payload.get("name", payload.get("clip", key))
            else:
                continue
            if key in REQUIRED_CLIPS and isinstance(actual, str) and actual.strip():
                result[key] = actual.strip()
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for payload in value:
            if isinstance(payload, str):
                name = payload.strip()
            elif isinstance(payload, Mapping):
                name = payload.get("name", payload.get("clip", ""))
                if not isinstance(name, str):
                    continue
                name = name.strip()
            else:
                continue
            canonical = name.lower()
            if canonical in REQUIRED_CLIPS:
                result[canonical] = name
        return result
    raise ValueError("character clips/animations must be a mapping or list")


def parse_character_manifest(data: Any) -> CharacterManifest:
    """Parse optional bounds, physics pivot, clips, and hand-joint names."""

    if not isinstance(data, Mapping):
        raise ValueError("character manifest root must be a JSON object")

    character = data.get("character")
    if not isinstance(character, Mapping):
        character = {}

    bounds_value: Any | None = None
    if "bounds" in data or "local_bounds_y_up" in data or "bounds_y_up" in data:
        bounds_value = data
    elif character:
        if any(
            key in character for key in ("bounds", "local_bounds_y_up", "bounds_y_up")
        ):
            bounds_value = character

    pivot_value = data.get("physics_pivot", data.get("pivot"))
    if pivot_value is None:
        pivot_value = character.get("physics_pivot", character.get("pivot"))
    pivot_y_up = data.get("physics_pivot_y_up")
    if pivot_y_up is None:
        pivot_y_up = character.get("physics_pivot_y_up")
    pivot_blender = data.get("physics_pivot_blender_z_up")
    if pivot_blender is None:
        pivot_blender = character.get("physics_pivot_blender_z_up")
    physics_pivot: tuple[float, float, float] | None = None
    if pivot_value is not None:
        physics_pivot = _vec3(pivot_value, "character.physics_pivot")
    elif pivot_y_up is not None:
        source_pivot = _vec3(pivot_y_up, "character.physics_pivot_y_up")
        physics_pivot = (
            source_pivot[0],
            -source_pivot[2],
            source_pivot[1],
        )
    elif pivot_blender is not None:
        source_pivot = _vec3(
            pivot_blender,
            "character.physics_pivot_blender_z_up",
        )
        physics_pivot = (
            source_pivot[0],
            -source_pivot[1],
            source_pivot[2],
        )

    heading_value = data.get("heading_offset_degrees")
    if heading_value is None:
        heading_value = character.get("heading_offset_degrees", 0.0)
    heading_offset_degrees = _number(
        heading_value,
        "character.heading_offset_degrees",
    )

    clips = _parse_clip_manifest(data.get("clips", data.get("animations")))
    raw_joints = data.get("joints", data.get("hand_joints", {}))
    joints: dict[str, str] = {}
    if raw_joints is not None:
        if not isinstance(raw_joints, Mapping):
            raise ValueError("character joints must be a mapping")
        aliases = {
            "right": ("right", "right_hand", "rightHand"),
            "left": ("left", "left_hand", "leftHand"),
        }
        for side, names in aliases.items():
            for name in names:
                candidate = raw_joints.get(name)
                if isinstance(candidate, Mapping):
                    candidate = candidate.get("name", candidate.get("joint"))
                if isinstance(candidate, str) and candidate.strip():
                    joints[side] = candidate.strip()
                    break

    return CharacterManifest(
        bounds=_bounds(bounds_value, "character") if bounds_value is not None else None,
        physics_pivot=physics_pivot,
        heading_offset_degrees=heading_offset_degrees,
        clips=clips,
        joints=joints,
    )


def load_character_manifest(path: str | Path) -> CharacterManifest:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise CharacterAssetError(f"character manifest not found: {manifest_path}")
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise CharacterAssetError(
            f"could not read character manifest {manifest_path}: {exc}"
        ) from exc
    try:
        return parse_character_manifest(data)
    except ValueError as exc:
        raise CharacterAssetError(
            f"invalid character manifest {manifest_path}: {exc}"
        ) from exc


def event_animation_requests(events: Iterable[Any]) -> tuple[CharacterAnimState, ...]:
    """Map edge-triggered SwingEvents without importing the physics module."""

    mapping = {
        "SHOT": CharacterAnimState.SHOOT,
        "ATTACH": CharacterAnimState.ATTACH,
        "RELEASE": CharacterAnimState.RELEASE,
    }
    requests = []
    for event in events:
        name = getattr(event, "name", str(event)).rsplit(".", 1)[-1].upper()
        request = mapping.get(name)
        if request is not None:
            requests.append(request)
    return tuple(requests)


def base_animation_for_sim(sim: Any) -> CharacterAnimState:
    """Choose the persistent pose after queued one-shots have completed."""

    attached = bool(getattr(sim, "attached", getattr(sim, "anchor", None) is not None))
    if attached:
        return CharacterAnimState.SWING
    if not bool(getattr(sim, "alive", True)):
        return CharacterAnimState.IDLE
    if float(getattr(sim, "elapsed", 0.0)) <= 0.0:
        return CharacterAnimState.IDLE
    return CharacterAnimState.FALL


def swing_phase(
    web_arc: float, max_arc: float = DEFAULT_SWING_ARC_RADIANS
) -> float:
    """Normalize the physics cable sweep for deterministic clip scrubbing."""

    try:
        arc = float(web_arc)
        limit = float(max_arc)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(arc) or not math.isfinite(limit) or limit <= 0.0:
        return 0.0
    return max(0.0, min(1.0, arc / limit))


def swing_frame(phase: float, frame_count: int) -> int:
    """Convert a normalized phase to a legal integer animation frame."""

    count = max(1, int(frame_count))
    try:
        value = float(phase)
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value):
        value = 0.0
    value = max(0.0, min(1.0, value))
    return int(round(value * (count - 1)))


def _normalized_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def resolve_clip_names(
    available: Iterable[str], manifest_clips: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Resolve required canonical clips against Actor animation names."""

    names = tuple(str(name) for name in available)
    normalized: dict[str, list[str]] = {}
    for name in names:
        normalized.setdefault(_normalized_name(name), []).append(name)

    requested = manifest_clips or {}
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for canonical in REQUIRED_CLIPS:
        wanted = str(requested.get(canonical, canonical))
        if wanted in names:
            resolved[canonical] = wanted
            continue
        matches = normalized.get(_normalized_name(wanted), [])
        if len(matches) == 1:
            resolved[canonical] = matches[0]
            continue
        # A few exporters prefix animation names with an armature name.
        suffix_matches = [
            name
            for name in names
            if any(
                name.lower().endswith(separator + wanted.lower())
                for separator in ("|", ":", "/")
            )
        ]
        if len(suffix_matches) == 1:
            resolved[canonical] = suffix_matches[0]
        else:
            missing.append(canonical)
    if missing:
        available_text = ", ".join(names) if names else "none"
        raise ValueError(
            "character is missing embedded clips "
            + ", ".join(missing)
            + f" (available: {available_text})"
        )
    return resolved


def select_hand_joint(
    joint_names: Iterable[str], side: str, manifest_name: str | None = None
) -> str | None:
    """Find a deform hand joint, preferring the manifest's exact socket."""

    side = side.lower()
    if side not in ("right", "left"):
        raise ValueError("hand side must be 'right' or 'left'")
    names = tuple(str(name) for name in joint_names)
    by_normalized: dict[str, list[str]] = {}
    for name in names:
        by_normalized.setdefault(_normalized_name(name), []).append(name)

    if manifest_name:
        if manifest_name in names:
            return manifest_name
        matches = by_normalized.get(_normalized_name(manifest_name), [])
        if len(matches) == 1:
            return matches[0]

    for alias in _HAND_ALIASES[side]:
        matches = by_normalized.get(_normalized_name(alias), [])
        if len(matches) == 1:
            return matches[0]

    reject = ("finger", "thumb", "index", "middle", "ring", "pinky", "ik")
    candidates: list[tuple[int, int, str]] = []
    for name in names:
        normalized = _normalized_name(name)
        if "hand" not in normalized or any(word in normalized for word in reject):
            continue
        side_match = (
            "right" in normalized
            or normalized.endswith("handr")
            or normalized.startswith("rhand")
        ) if side == "right" else (
            "left" in normalized
            or normalized.endswith("handl")
            or normalized.startswith("lhand")
        )
        if side_match:
            # Prefer DEF/deform bones over controls, then shorter names.
            deform_rank = 0 if normalized.startswith("def") else 1
            candidates.append((deform_rank, len(normalized), name))
    return min(candidates)[2] if candidates else None


def _method(value: Any, snake_name: str, camel_name: str) -> Any:
    method = getattr(value, snake_name, None)
    if method is None:
        method = getattr(value, camel_name)
    return method


def _load_actor(asset_path: Path) -> Any:
    try:
        from direct.actor.Actor import Actor
        from panda3d.core import NodePath
    except (ImportError, ModuleNotFoundError) as exc:
        raise Panda3DUnavailableError(
            "Panda3D is required for CharacterController; install panda3d "
            "and panda3d-gltf"
        ) from exc

    try:
        # The direct panda3d-gltf entry point works headlessly and preserves
        # embedded Character/AnimBundle nodes for Actor's auto-binding.
        from gltf import load_model

        model = NodePath(load_model(str(asset_path.resolve())))
        return Actor(model, copy=False)
    except (ImportError, ModuleNotFoundError):
        # Actor initializes Panda's registered Python file types before loading.
        return Actor(str(asset_path.resolve()))


def _actor_bounds(actor: Any) -> CharacterBounds:
    result = _method(actor, "get_tight_bounds", "getTightBounds")()
    if not result or len(result) != 2 or result[0] is None or result[1] is None:
        raise CharacterAssetError("character model has no measurable geometry bounds")
    try:
        return CharacterBounds(
            tuple(float(value) for value in result[0]),  # type: ignore[arg-type]
            tuple(float(value) for value in result[1]),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise CharacterAssetError("character model returned invalid geometry bounds") from exc


class CharacterController:
    """Scale, pivot, animate, and expose hand sockets for a Panda3D Actor."""

    def __init__(
        self,
        parent: Any,
        asset_path: str | Path,
        target_height: float = 4.8,
        manifest_path: str | Path | None = None,
    ) -> None:
        self.asset_path = Path(asset_path)
        try:
            self.target_height = float(target_height)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_height must be numeric") from exc
        if not math.isfinite(self.target_height) or self.target_height <= 0.0:
            raise ValueError("target_height must be finite and positive")
        if not self.asset_path.is_file():
            raise CharacterAssetError(f"character GLB asset not found: {self.asset_path}")
        if parent is None or not (
            hasattr(parent, "attach_new_node") or hasattr(parent, "attachNewNode")
        ):
            raise TypeError("parent must be a Panda3D NodePath")

        explicit_manifest = manifest_path is not None
        if manifest_path is None:
            for filename in ("character_manifest.json", "manifest.json"):
                adjacent = self.asset_path.with_name(filename)
                if adjacent.is_file():
                    manifest_path = adjacent
                    break
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        if explicit_manifest or self.manifest_path is not None:
            assert self.manifest_path is not None
            self.manifest = load_character_manifest(self.manifest_path)
        else:
            self.manifest = CharacterManifest()

        self.parent = parent
        self.root = _method(parent, "attach_new_node", "attachNewNode")(
            "character-visual-root"
        )
        if abs(self.manifest.heading_offset_degrees) > 1e-9:
            _method(self.root, "set_h", "setH")(
                self.manifest.heading_offset_degrees
            )
        self.actor: Any | None = None
        self._cleaned = False
        try:
            try:
                self.actor = _load_actor(self.asset_path)
            except (Panda3DUnavailableError, CharacterAssetError):
                raise
            except Exception as exc:
                raise CharacterAssetError(
                    f"could not load character GLB {self.asset_path}: {exc}"
                ) from exc
            _method(self.actor, "reparent_to", "reparentTo")(self.root)

            available = _method(
                self.actor, "get_anim_names", "getAnimNames"
            )()
            try:
                self.clips = resolve_clip_names(available, self.manifest.clips)
            except ValueError as exc:
                raise CharacterAssetError(str(exc)) from exc

            # Establish a stable reference pose before measuring fallback bounds.
            _method(self.actor, "pose", "pose")(self.clips["idle"], 0)
            bounds = self.manifest.bounds or _actor_bounds(self.actor)
            self.model_scale = self.target_height / bounds.height
            pivot = self.manifest.physics_pivot or tuple(
                (lower + upper) * 0.5
                for lower, upper in zip(bounds.minimum, bounds.maximum)
            )
            _method(self.actor, "set_scale", "setScale")(self.model_scale)
            _method(self.actor, "set_pos", "setPos")(
                -pivot[0] * self.model_scale,
                -pivot[1] * self.model_scale,
                -pivot[2] * self.model_scale,
            )

            self._fallback_hands = self._create_fallback_hands()
            self.hand_joint_names, self._hand_nodes = self._expose_hands()

            # Empty sentinels ensure the initial idle loop is actually started.
            self.state: CharacterAnimState | None = None
            self.current_clip = ""
            self._active_one_shot: CharacterAnimState | None = None
            self._one_shot_remaining = 0.0
            self._one_shot_queue: deque[CharacterAnimState] = deque()
            self._last_swing_frame: int | None = None
            self._loop(CharacterAnimState.IDLE)
        except Exception:
            self.cleanup()
            raise

    def _create_fallback_hands(self) -> dict[str, Any]:
        hands: dict[str, Any] = {}
        for side, sign in (("right", 1.0), ("left", -1.0)):
            node = _method(self.root, "attach_new_node", "attachNewNode")(
                f"fallback-{side}-hand"
            )
            _method(node, "set_pos", "setPos")(
                sign * self.target_height * 0.24,
                self.target_height * 0.06,
                self.target_height * 0.15,
            )
            hands[side] = node
        return hands

    def _expose_hands(self) -> tuple[dict[str, str | None], dict[str, Any]]:
        assert self.actor is not None
        try:
            joints = _method(self.actor, "get_joints", "getJoints")(
                "modelRoot", "*"
            )
            joint_names = [
                str(_method(joint, "get_name", "getName")()) for joint in joints
            ]
        except Exception:
            joint_names = []

        selected: dict[str, str | None] = {}
        nodes: dict[str, Any] = {}
        for side in ("right", "left"):
            name = select_hand_joint(
                joint_names, side, self.manifest.joints.get(side)
            )
            selected[side] = name
            if name is None:
                nodes[side] = self._fallback_hands[side]
                continue
            try:
                exposed = _method(self.actor, "expose_joint", "exposeJoint")(
                    None, "modelRoot", name
                )
            except Exception:
                exposed = None
            nodes[side] = exposed or self._fallback_hands[side]
        return selected, nodes

    def _actual(self, state: CharacterAnimState) -> str:
        return self.clips[state.value]

    def _loop(self, state: CharacterAnimState) -> None:
        if self.current_clip == state.value and self.state == state:
            return
        assert self.actor is not None
        _method(self.actor, "stop", "stop")()
        _method(self.actor, "set_play_rate", "setPlayRate")(
            1.0, self._actual(state)
        )
        _method(self.actor, "loop", "loop")(self._actual(state), restart=1)
        self.state = state
        self.current_clip = state.value
        self._last_swing_frame = None

    def _start_one_shot(self, state: CharacterAnimState) -> None:
        assert self.actor is not None
        actual = self._actual(state)
        target_duration = ONE_SHOT_SECONDS[state.value]
        try:
            source_duration = float(
                _method(self.actor, "get_duration", "getDuration")(actual)
            )
        except (TypeError, ValueError):
            source_duration = target_duration
        if not math.isfinite(source_duration) or source_duration <= 0.0:
            source_duration = target_duration
        _method(self.actor, "stop", "stop")()
        _method(self.actor, "set_play_rate", "setPlayRate")(
            source_duration / target_duration, actual
        )
        _method(self.actor, "play", "play")(actual)
        self._active_one_shot = state
        self._one_shot_remaining = target_duration
        self.state = state
        self.current_clip = state.value
        self._last_swing_frame = None

    def _queue_events(self, events: Iterable[Any]) -> None:
        for request in event_animation_requests(events):
            if request == CharacterAnimState.RELEASE:
                # Release is player-critical feedback and must not wait behind
                # a stale shoot/attach clip.
                self._one_shot_queue.clear()
                self._start_one_shot(request)
            elif self._active_one_shot is None:
                self._start_one_shot(request)
            else:
                self._one_shot_queue.append(request)

    def _advance_one_shots(self, dt: float) -> bool:
        remaining_dt = max(0.0, dt)
        while self._active_one_shot is not None:
            if remaining_dt < self._one_shot_remaining:
                self._one_shot_remaining -= remaining_dt
                return True
            remaining_dt -= self._one_shot_remaining
            self._active_one_shot = None
            self._one_shot_remaining = 0.0
            if self._one_shot_queue:
                self._start_one_shot(self._one_shot_queue.popleft())
            else:
                return False
        return False

    def _apply_swing_pose(self, sim: Any) -> None:
        assert self.actor is not None
        actual = self._actual(CharacterAnimState.SWING)
        try:
            count = int(
                _method(self.actor, "get_num_frames", "getNumFrames")(actual)
            )
        except (TypeError, ValueError):
            count = 1
        frame = swing_frame(swing_phase(getattr(sim, "web_arc", 0.0)), count)
        if self.state != CharacterAnimState.SWING:
            _method(self.actor, "stop", "stop")()
        if frame != self._last_swing_frame or self.state != CharacterAnimState.SWING:
            _method(self.actor, "pose", "pose")(actual, frame)
            self._last_swing_frame = frame
        self.state = CharacterAnimState.SWING
        self.current_clip = "swing"

    def update(self, sim: Any, events: Iterable[Any], dt: float) -> None:
        """Advance event one-shots, then apply the physics-derived base pose."""

        if self._cleaned:
            raise CharacterControllerError("character controller has been cleaned up")
        try:
            step = float(dt)
        except (TypeError, ValueError):
            step = 0.0
        if not math.isfinite(step) or step < 0.0:
            step = 0.0

        self._queue_events(events)
        if self._advance_one_shots(step):
            return

        base = base_animation_for_sim(sim)
        if base == CharacterAnimState.SWING:
            self._apply_swing_pose(sim)
        else:
            self._loop(base)

    def reset(
        self,
        state: CharacterAnimState | str = CharacterAnimState.IDLE,
    ) -> None:
        """Clear transient clips and immediately enter a deliberate base loop."""

        if self._cleaned:
            raise CharacterControllerError("character controller has been cleaned up")
        try:
            requested = (
                state if isinstance(state, CharacterAnimState) else CharacterAnimState(state)
            )
        except ValueError as exc:
            raise ValueError(f"unknown character animation state: {state}") from exc
        if requested in {
            CharacterAnimState.SHOOT,
            CharacterAnimState.ATTACH,
            CharacterAnimState.RELEASE,
        }:
            raise ValueError("reset state must be idle, fall, or swing")
        self._one_shot_queue.clear()
        self._active_one_shot = None
        self._one_shot_remaining = 0.0
        self.current_clip = ""
        self.state = None
        self._last_swing_frame = None
        if requested == CharacterAnimState.SWING:
            assert self.actor is not None
            actual = self._actual(requested)
            _method(self.actor, "stop", "stop")()
            _method(self.actor, "pose", "pose")(actual, 0)
            self.state = requested
            self.current_clip = requested.value
            self._last_swing_frame = 0
        else:
            self._loop(requested)

    def _anchor_side(self, anchor: Any | None) -> str:
        if anchor is None:
            return "right"
        try:
            anchor_x = float(anchor.x)
        except (AttributeError, TypeError, ValueError):
            try:
                anchor_x = float(anchor[0])
            except (TypeError, ValueError, IndexError):
                return "right"

        # Rig labels are anatomical, so a forward-facing character's `.L`
        # joint can live on positive world X.  Select by the actual animated
        # wrist position instead of assuming naming and screen side agree.
        try:
            top = _method(self.parent, "get_top", "getTop")()
            candidates = {
                side: float(
                    _method(node, "get_pos", "getPos")(top)[0]
                )
                for side, node in self._hand_nodes.items()
            }
            if candidates:
                return min(candidates, key=lambda side: abs(candidates[side] - anchor_x))
        except Exception:
            pass
        try:
            top = _method(self.parent, "get_top", "getTop")()
            parent_x = float(_method(self.parent, "get_x", "getX")(top))
        except Exception:
            parent_x = 0.0
        return "left" if anchor_x < parent_x else "right"

    def hand_world_position(self, anchor: Any | None = None) -> tuple[float, float, float]:
        """Return the selected animated hand in Panda3D world coordinates."""

        if self._cleaned or self.actor is None:
            raise CharacterControllerError("character controller has been cleaned up")
        try:
            _method(self.actor, "update", "update")(force=True)
        except Exception:
            pass
        side = self._anchor_side(anchor)
        node = self._hand_nodes.get(side, self._fallback_hands[side])
        top = _method(self.parent, "get_top", "getTop")()
        point = _method(node, "get_pos", "getPos")(top)
        return tuple(float(value) for value in point)  # type: ignore[return-value]

    def cleanup(self) -> None:
        """Release Actor animation resources without removing the caller parent."""

        if self._cleaned:
            return
        self._cleaned = True
        actor = self.actor
        self.actor = None
        if actor is not None:
            try:
                _method(actor, "cleanup", "cleanup")()
            except Exception:
                pass
        root = getattr(self, "root", None)
        if root is not None:
            try:
                _method(root, "remove_node", "removeNode")()
            except Exception:
                pass


__all__ = [
    "CharacterAnimState",
    "CharacterAssetError",
    "CharacterBounds",
    "CharacterController",
    "CharacterControllerError",
    "CharacterManifest",
    "Panda3DUnavailableError",
    "base_animation_for_sim",
    "event_animation_requests",
    "load_character_manifest",
    "parse_character_manifest",
    "resolve_clip_names",
    "select_hand_joint",
    "swing_frame",
    "swing_phase",
]
