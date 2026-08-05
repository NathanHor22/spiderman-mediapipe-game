"""Panda3D presentation adapter for the logical :mod:`render.world` strip.

``WorldStrip`` remains the authority for building footprints, roofs, collision,
and web anchors.  This module only mirrors its short-lived ``Building`` proxies
with instances from a GLB library.  In particular, no dimensions measured from
the rendered model are fed back into gameplay.

The manifest bounds are local bounds for each named GLB node.  Native Panda3D
bounds use X for street width, Y for forward/depth, and Z for up.  Exporter
bounds marked ``local_bounds_y_up`` are converted from glTF's X/right, Y/up,
Z/back convention.  The canonical Panda-space manifest shape is::

    {
      "variants": {
        "b1": {"bounds": {"min": [-1, -1, 0], "max": [1, 1, 6]}},
        "b2": {"bounds": {"min": [-1, -1, 0], "max": [1, 1, 8]}}
      }
    }

``nodes``, ``buildings``, or a list of named records are accepted as well so
the exporter does not have to duplicate data merely to satisfy this adapter.
Panda3D is imported lazily; the pure manifest/fit helpers remain usable in
tools and unit tests that do not install the 3D runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_VARIANT_NAMES = tuple(
    [f"b{index}" for index in range(1, 7)]
    + [f"g{index}" for index in range(1, 7)]
)


class BuildingRendererError(RuntimeError):
    """Base class for actionable building presentation failures."""


class BuildingAssetError(BuildingRendererError):
    """The GLB or its manifest is absent, malformed, or inconsistent."""


class Panda3DUnavailableError(BuildingRendererError):
    """The optional Panda3D presentation runtime is not installed."""


@dataclass(frozen=True)
class Bounds3D:
    """An axis-aligned local-space box in Panda3D X/Y/Z order."""

    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = self.minimum + self.maximum
        if not all(math.isfinite(value) for value in values):
            raise ValueError("building bounds must contain only finite numbers")
        if any(hi <= lo for lo, hi in zip(self.minimum, self.maximum)):
            raise ValueError(
                "building bounds must have max greater than min on every axis"
            )

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(
            hi - lo for lo, hi in zip(self.minimum, self.maximum)
        )  # type: ignore[return-value]


@dataclass(frozen=True)
class VariantSpec:
    """A named instancing source and the bounds authored for that node."""

    name: str
    bounds: Bounds3D


@dataclass(frozen=True)
class FitTransform:
    """Panda3D position and scale that map a variant onto a logical proxy."""

    position: tuple[float, float, float]
    scale: tuple[float, float, float]


@dataclass
class _RenderedBuilding:
    holder: Any
    variant_name: str


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


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


def _parse_bounds(value: Any, label: str) -> Bounds3D:
    # Prefer the exporter's canonical Panda-space bounds when both forms are
    # present.  The Y-up branch remains a compatibility path for older assets.
    if isinstance(value, Mapping) and "bounds" in value:
        return _parse_bounds(value["bounds"], f"{label}.bounds")

    if isinstance(value, Mapping):
        y_up = value.get("local_bounds_y_up", value.get("bounds_y_up"))
        if y_up is not None:
            source = _parse_bounds(y_up, f"{label}.local_bounds_y_up")
            # glTF forward is -Z; Panda3D forward is +Y.  Negating an AABB
            # swaps its old lower/upper Z endpoints.
            return Bounds3D(
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

    if isinstance(value, Mapping):
        lower = value.get("min", value.get("minimum"))
        upper = value.get("max", value.get("maximum"))
        if lower is not None and upper is not None:
            return Bounds3D(
                _vec3(lower, f"{label}.min"),
                _vec3(upper, f"{label}.max"),
            )

        if all(axis in value for axis in ("x", "y", "z")):
            pairs = []
            for axis in ("x", "y", "z"):
                pair = value[axis]
                if (
                    isinstance(pair, (str, bytes))
                    or not isinstance(pair, Sequence)
                    or len(pair) != 2
                ):
                    raise ValueError(
                        f"{label}.{axis} must be a [min, max] pair"
                    )
                pairs.append(
                    (
                        _number(pair[0], f"{label}.{axis}[0]"),
                        _number(pair[1], f"{label}.{axis}[1]"),
                    )
                )
            return Bounds3D(
                tuple(pair[0] for pair in pairs),  # type: ignore[arg-type]
                tuple(pair[1] for pair in pairs),  # type: ignore[arg-type]
            )

    if (
        not isinstance(value, (str, bytes))
        and isinstance(value, Sequence)
        and len(value) == 2
    ):
        return Bounds3D(
            _vec3(value[0], f"{label}[0]"),
            _vec3(value[1], f"{label}[1]"),
        )

    raise ValueError(
        f"{label} must provide bounds as min/max vectors or two vectors"
    )


def _manifest_entries(data: Any) -> list[tuple[str, Any]]:
    if not isinstance(data, Mapping):
        raise ValueError("manifest root must be a JSON object")

    container: Any | None = None
    for key in ("variants", "nodes", "buildings", "models"):
        candidate = data.get(key)
        if isinstance(candidate, (Mapping, list)):
            container = candidate
            break

    # Some exporters naturally emit {"bounds": {"b1": ..., ...}}.
    if container is None:
        bounds_map = data.get("bounds")
        if isinstance(bounds_map, Mapping) and any(
            str(name).lower() in EXPECTED_VARIANT_NAMES for name in bounds_map
        ):
            container = bounds_map

    # A direct {"b1": {...}, "b2": {...}} map is also unambiguous.
    if container is None and any(
        str(name).lower() in EXPECTED_VARIANT_NAMES for name in data
    ):
        container = data

    if container is None:
        raise ValueError(
            "manifest must contain variants, nodes, buildings, or models"
        )

    if isinstance(container, Mapping):
        return [(str(name), payload) for name, payload in container.items()]

    entries: list[tuple[str, Any]] = []
    for index, payload in enumerate(container):
        if not isinstance(payload, Mapping):
            raise ValueError(f"manifest entry {index} must be an object")
        name = payload.get("name", payload.get("node", payload.get("id")))
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"manifest entry {index} must have a non-empty name or node"
            )
        entries.append((name, payload))
    return entries


def parse_manifest_data(
    data: Any, *, require_expected: bool = True
) -> tuple[VariantSpec, ...]:
    """Parse exporter JSON into validated, predictably ordered variants.

    ``require_expected=False`` is useful for asset-pipeline tests and preview
    tools.  The runtime renderer uses the strict default because a partially
    exported library otherwise fails much later and much less clearly.
    """

    variants: dict[str, VariantSpec] = {}
    for raw_name, payload in _manifest_entries(data):
        name = raw_name.strip().lower()
        # Metadata alongside a direct map is harmless and is not a variant.
        if name not in EXPECTED_VARIANT_NAMES:
            continue
        if name in variants:
            raise ValueError(f"manifest contains duplicate variant {name!r}")
        variants[name] = VariantSpec(name, _parse_bounds(payload, name))

    if require_expected:
        missing = [name for name in EXPECTED_VARIANT_NAMES if name not in variants]
        if missing:
            raise ValueError(
                "manifest is missing required building variants: "
                + ", ".join(missing)
            )
    elif not variants:
        raise ValueError("manifest does not contain any b1-b6/g1-g6 variants")

    return tuple(
        variants[name] for name in EXPECTED_VARIANT_NAMES if name in variants
    )


def load_manifest(
    manifest_path: str | Path, *, require_expected: bool = True
) -> tuple[VariantSpec, ...]:
    """Load a building manifest with errors that identify the bad asset."""

    path = Path(manifest_path)
    if not path.is_file():
        raise BuildingAssetError(f"building manifest not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildingAssetError(
            f"could not read building manifest {path}: {exc}"
        ) from exc
    try:
        return parse_manifest_data(data, require_expected=require_expected)
    except ValueError as exc:
        raise BuildingAssetError(f"invalid building manifest {path}: {exc}") from exc


def proxy_dimensions(proxy: Any) -> tuple[float, float, float]:
    """Return logical width/depth/height in Panda3D axis order."""

    try:
        width = abs(float(proxy.x1) - float(proxy.x0))
        depth = abs(float(proxy.z1) - float(proxy.z0))
        height = float(proxy.height)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "building proxy must provide numeric x0, x1, z0, z1, and height"
        ) from exc
    dimensions = (width, depth, height)
    if not all(math.isfinite(value) and value > 0.0 for value in dimensions):
        raise ValueError("building proxy dimensions must be finite and positive")
    return dimensions


def proxy_key(proxy: Any) -> tuple[int, float, float, float, float, float]:
    """Stable key for an immutable logical building proxy.

    WorldStrip retains proxy objects until retirement, but using geometry rather
    than ``id(proxy)`` also avoids needless instance churn if a replay restores
    equivalent proxies from serialized state.
    """

    # Validate dimensions first so a malformed proxy cannot enter the cache.
    proxy_dimensions(proxy)
    try:
        raw_side = float(proxy.side)
        values = tuple(
            float(getattr(proxy, field))
            for field in ("x0", "x1", "z0", "z1", "height")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("building proxy must provide a numeric side") from exc
    if not math.isfinite(raw_side) or raw_side not in (-1.0, 1.0):
        raise ValueError("building proxy side must be -1 or +1")
    side = int(raw_side)
    return (side, *values)


def _shape_distortion(
    source: Sequence[float], target: Sequence[float]
) -> float:
    # Exact per-axis fitting uses these three scale factors.  Their spread in
    # log space is zero for a uniform (aspect-preserving) scale and treats a 2x
    # stretch the same as a 1/2x squeeze.
    logs = [math.log(dst / src) for src, dst in zip(source, target)]
    mean = sum(logs) / len(logs)
    return sum((value - mean) ** 2 for value in logs)


def _stable_tie_break(seed: int, key: Any, name: str) -> int:
    token = f"{seed!r}|{key!r}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "big")


def choose_variant(
    variants: Iterable[VariantSpec],
    target_size: Sequence[float],
    *,
    seed: int = 0,
    key: Any = (),
) -> VariantSpec:
    """Choose the least-distorted variant with a stable seeded tie-break.

    Selection is independent of manifest and world-list order, so retiring a
    building does not reshuffle every model ahead of it.
    """

    target = _vec3(target_size, "target_size")
    if any(value <= 0.0 for value in target):
        raise ValueError("target_size values must be positive")
    candidates = tuple(variants)
    if not candidates:
        raise ValueError("at least one building variant is required")
    for variant in candidates:
        if not isinstance(variant, VariantSpec):
            raise TypeError("variants must contain VariantSpec values")
    return min(
        candidates,
        key=lambda variant: (
            round(_shape_distortion(variant.bounds.size, target), 12),
            _stable_tie_break(seed, key, variant.name),
            variant.name,
        ),
    )


def fit_variant_to_proxy(variant: VariantSpec, proxy: Any) -> FitTransform:
    """Fit a chosen node exactly to a WorldStrip footprint and roof.

    The preceding shape-aware selection minimizes anisotropic scaling.  Exact
    fitting here keeps the rendered wall and roof aligned with the logical web
    anchors; allowing a uniformly scaled mesh to overhang would make a web
    visibly attach to empty space.
    """

    target_size = proxy_dimensions(proxy)
    try:
        target_min = (
            min(float(proxy.x0), float(proxy.x1)),
            min(float(proxy.z0), float(proxy.z1)),
            0.0,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("building proxy has invalid footprint coordinates") from exc

    source_min = variant.bounds.minimum
    source_size = variant.bounds.size
    scale = tuple(
        target / source for target, source in zip(target_size, source_size)
    )
    position = tuple(
        target - source * factor
        for target, source, factor in zip(target_min, source_min, scale)
    )
    return FitTransform(
        position=position,  # type: ignore[arg-type]
        scale=scale,  # type: ignore[arg-type]
    )


def _panda_method(value: Any, snake_name: str, camel_name: str) -> Any:
    method = getattr(value, snake_name, None)
    if method is None:
        method = getattr(value, camel_name)
    return method


def _import_panda() -> tuple[Any, Any, Any, Any]:
    try:
        from panda3d.core import Filename, Loader, LoaderOptions, NodePath
    except (ImportError, ModuleNotFoundError) as exc:
        raise Panda3DUnavailableError(
            "Panda3D is required for BuildingRenderer; install panda3d and "
            "panda3d-gltf to load the GLB building library"
        ) from exc
    return Filename, Loader, LoaderOptions, NodePath


def _load_panda_model(
    asset_path: Path,
    Filename: Any,
    Loader: Any,
    LoaderOptions: Any,
) -> Any:
    """Load GLB through panda3d-gltf, falling back to Panda's registry.

    Calling ``panda3d.core.Loader`` directly does not reliably register the
    Python glTF loader.  On installations that also include Assimp it may even
    select that plugin and fail on an otherwise valid GLB.  ``gltf.load_model``
    is the package's supported headless entry point and needs no ``ShowBase``.
    """

    try:
        from gltf import load_model
    except (ImportError, ModuleNotFoundError):
        filename = _panda_method(
            Filename, "from_os_specific", "fromOsSpecific"
        )(str(asset_path.resolve()))
        loader = _panda_method(Loader, "get_global_ptr", "getGlobalPtr")()
        return _panda_method(loader, "load_sync", "loadSync")(
            filename, LoaderOptions()
        )
    return load_model(str(asset_path.resolve()))


class BuildingRenderer:
    """Mirror ``world.buildings`` using reusable nodes from a GLB library."""

    def __init__(
        self,
        parent: Any,
        asset_path: str | Path,
        manifest_path: str | Path,
        seed: int = 7,
    ) -> None:
        self.asset_path = Path(asset_path)
        self.manifest_path = Path(manifest_path)
        self.seed = int(seed)

        if not self.asset_path.is_file():
            raise BuildingAssetError(
                f"building GLB asset not found: {self.asset_path}"
            )
        self.variants = load_manifest(self.manifest_path, require_expected=True)

        Filename, Loader, LoaderOptions, NodePath = _import_panda()
        if parent is None or not (
            hasattr(parent, "attach_new_node") or hasattr(parent, "attachNewNode")
        ):
            raise TypeError("parent must be a Panda3D NodePath")

        self.root = _panda_method(
            parent, "attach_new_node", "attachNewNode"
        )("world-buildings")
        self._instances: dict[
            tuple[int, float, float, float, float, float], _RenderedBuilding
        ] = {}

        try:
            loaded = _load_panda_model(
                self.asset_path, Filename, Loader, LoaderOptions
            )
        except Exception as exc:
            _panda_method(self.root, "remove_node", "removeNode")()
            raise BuildingAssetError(
                f"could not load building GLB {self.asset_path}: {exc}"
            ) from exc

        if loaded is None:
            _panda_method(self.root, "remove_node", "removeNode")()
            raise BuildingAssetError(
                f"Panda3D could not load building GLB {self.asset_path}; "
                "check that panda3d-gltf is installed and the file is valid"
            )
        self._asset_root = loaded if hasattr(loaded, "find") else NodePath(loaded)
        if _panda_method(self._asset_root, "is_empty", "isEmpty")():
            _panda_method(self.root, "remove_node", "removeNode")()
            raise BuildingAssetError(
                f"Panda3D loaded an empty building asset: {self.asset_path}"
            )

        # Templates are detached from the render graph.  Each is copied only
        # once from the GLB, then true-instanced for every logical building.
        self._template_root = NodePath("building-variant-templates")
        self._templates: dict[str, Any] = {}
        try:
            for variant in self.variants:
                source = self._asset_root.find(f"**/{variant.name}")
                if _panda_method(source, "is_empty", "isEmpty")():
                    raise BuildingAssetError(
                        f"building GLB {self.asset_path} has no node named "
                        f"{variant.name!r}"
                    )
                template = _panda_method(source, "copy_to", "copyTo")(
                    self._template_root
                )
                # The node's local origin/bounds, not an asset-library layout
                # offset, is what the manifest describes.
                _panda_method(template, "clear_transform", "clearTransform")()
                self._templates[variant.name] = template
        except Exception:
            _panda_method(self.root, "remove_node", "removeNode")()
            raise

    @property
    def instance_count(self) -> int:
        return len(self._instances)

    def sync(self, world: Any) -> None:
        """Create and retire render instances to match ``world.buildings``."""

        try:
            proxies = tuple(world.buildings)
        except (AttributeError, TypeError) as exc:
            raise TypeError("world must provide an iterable buildings collection") from exc

        desired: dict[
            tuple[int, float, float, float, float, float],
            tuple[VariantSpec, FitTransform],
        ] = {}
        for proxy in proxies:
            key = proxy_key(proxy)
            if key in desired:
                raise BuildingRendererError(
                    "world contains duplicate building proxies with the same footprint"
                )
            variant = choose_variant(
                self.variants,
                proxy_dimensions(proxy),
                seed=self.seed,
                key=key,
            )
            desired[key] = (variant, fit_variant_to_proxy(variant, proxy))

        for key in tuple(self._instances):
            if key not in desired:
                rendered = self._instances.pop(key)
                _panda_method(rendered.holder, "remove_node", "removeNode")()

        for key in sorted(desired):
            if key in self._instances:
                continue
            variant, transform = desired[key]
            digest = hashlib.blake2b(repr(key).encode(), digest_size=4).hexdigest()
            holder = _panda_method(
                self.root, "attach_new_node", "attachNewNode"
            )(f"building-{variant.name}-{digest}")
            try:
                _panda_method(holder, "set_pos", "setPos")(*transform.position)
                _panda_method(holder, "set_scale", "setScale")(*transform.scale)
                _panda_method(holder, "set_tag", "setTag")(
                    "building_variant", variant.name
                )
                _panda_method(
                    self._templates[variant.name], "instance_to", "instanceTo"
                )(holder)
            except Exception:
                _panda_method(holder, "remove_node", "removeNode")()
                raise
            self._instances[key] = _RenderedBuilding(holder, variant.name)

    def clear(self) -> None:
        """Remove all live world instances while retaining loaded templates."""

        for rendered in self._instances.values():
            _panda_method(rendered.holder, "remove_node", "removeNode")()
        self._instances.clear()
