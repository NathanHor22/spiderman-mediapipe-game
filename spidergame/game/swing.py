"""Constrained 3D web-swinging physics.

The player is a particle and an attached web is a unilateral distance
constraint: it may pull when taut, but it never pushes when slack. Gravity is
always applied. Its component along the cable is cancelled by tension while
its tangential component changes the swing's angular velocity, producing a
real pendulum arc rather than a bungee response.

The state stays Cartesian because anchors sit above, beside and ahead of the
player. A single planar angle cannot describe that motion. The equivalent 3D
angular velocity is derived as ``cross(radius, velocity) / radius**2``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

from ..control import ControlState
from ..render.world import WorldStrip
from . import tuning as T


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


class SwingEvent(Enum):
    SHOT = auto()
    ATTACH = auto()
    MISS = auto()
    RELEASE = auto()


@dataclass
class Anchor:
    x: float
    y: float
    z: float
    rest_length: float
    taut: bool = True
    reel_target: float = 0.0


class SwingSim:
    def __init__(self, start_z: float = 0.0) -> None:
        self.reset(start_z)

    def reset(self, start_z: float = 0.0) -> None:
        self.x = 0.0
        self.y = T.START_Y
        self.z = start_z
        self.vx = 0.0
        self.vy = 0.0
        self.vz = T.START_SPEED
        self.anchor: Anchor | None = None
        self._armed = True
        self.elapsed = 0.0
        self.alive = True
        self.death_reason = ""

        self.whiffs = 0
        self.attaches = 0
        self.whiff_reason = {"no building": 0, "too low": 0, "out of range": 0}
        self.peak_y = self.y
        self.low_y = self.y

        # Derived cable state exposed for diagnostics and regression tests.
        self.angular_velocity = (0.0, 0.0, 0.0)
        self.web_tension = 0.0
        self.web_age = 0.0
        self.web_arc = 0.0
        self.release_reason = ""
        self.shots = 0
        self.auto_releases = 0
        self.last_events: tuple[SwingEvent, ...] = ()
        self._frame_events: list[SwingEvent] = []
        self._previous_rope_direction: tuple[float, float, float] | None = None

    # ---------------------------------------------------------------- web

    def _pick_anchor(self, world: WorldStrip, ctrl: ControlState) -> Anchor | None:
        """Choose a high, forward anchor or record why the shot missed."""
        side = -1 if ctrl.hand_x < 0.5 else 1

        offset = abs(ctrl.hand_x - 0.5) * 2.0
        ahead = (
            T.ANCHOR_AHEAD_MIN
            + (T.ANCHOR_AHEAD_MAX - T.ANCHOR_AHEAD_MIN) * offset
        )
        target_z = self.z + ahead

        def footprint_distance(building) -> float:
            if building.z0 <= target_z <= building.z1:
                return 0.0
            return min(abs(target_z - building.z0), abs(target_z - building.z1))

        candidates = [
            building
            for building in world.buildings
            if (
                building.side == side
                and building.z1 >= self.z + T.ANCHOR_AHEAD_MIN
                and footprint_distance(building) <= T.ANCHOR_SEARCH_WINDOW
            )
        ]
        candidates.sort(key=lambda building: (
            footprint_distance(building),
            -building.height,
        ))
        if not candidates:
            self.whiff_reason["no building"] += 1
            return None

        # A physically accurate line cannot lift from a shallow anchor. Clamp
        # low hand aiming upward while preserving the hand's higher choices.
        want_y = (
            T.ANCHOR_HEIGHT_MAX
            + (T.ANCHOR_HEIGHT_MIN - T.ANCHOR_HEIGHT_MAX) * ctrl.hand_y
        )
        slope = T.MIN_ANCHOR_UP_DOT / math.sqrt(
            1.0 - T.MIN_ANCHOR_UP_DOT ** 2
        )
        saw_too_low = False
        saw_out_of_range = False
        for building in candidates:
            az = max(building.z0, min(target_z, building.z1))
            ax = building.inner_x
            horizontal = math.hypot(ax - self.x, az - self.z)
            min_rise = max(T.ANCHOR_MIN_RISE, horizontal * slope)
            min_y = self.y + min_rise
            roof_y = building.height - T.ANCHOR_MIN_CLEARANCE
            if roof_y < min_y:
                saw_too_low = True
                continue

            ay = min(max(want_y, min_y), roof_y)
            dist = math.dist((ax, ay, az), (self.x, self.y, self.z))
            if dist > T.MAX_WEB_RANGE:
                saw_out_of_range = True
                continue

            # Start at the real distance. A shorter hard constraint would
            # teleport the character; the catch supplies the initial yank.
            return Anchor(ax, ay, az, rest_length=max(T.MIN_REST, dist))

        if saw_out_of_range:
            self.whiff_reason["out of range"] += 1
        elif saw_too_low:
            self.whiff_reason["too low"] += 1
        else:
            self.whiff_reason["no building"] += 1
        return None

    def _catch_web(self, anchor: Anchor) -> None:
        """Apply a bounded impulse along the cable with a known upward part."""
        radius = (
            self.x - anchor.x,
            self.y - anchor.y,
            self.z - anchor.z,
        )
        dist = math.sqrt(_dot(radius, radius))
        if dist < 1e-6:
            return

        outward = tuple(component / dist for component in radius)
        upward_fraction = max(1e-6, -outward[1])
        catch_dv = min(
            T.WEB_CATCH_MAX_DV,
            T.WEB_CATCH_UP_SPEED / upward_fraction,
        )
        self.vx -= outward[0] * catch_dv
        self.vy -= outward[1] * catch_dv
        self.vz -= outward[2] * catch_dv
        anchor.taut = True
        anchor.reel_target = max(
            T.MIN_REST,
            anchor.rest_length - T.WEB_REEL_DISTANCE,
        )
        self.web_age = 0.0
        self.web_arc = 0.0
        self.release_reason = ""
        self._previous_rope_direction = outward

    def _detach_web(self, reason: str, *, automatic: bool = False) -> None:
        """Release without changing velocity, preserving the launch impulse."""
        if self.anchor is None:
            return
        self.anchor = None
        self.angular_velocity = (0.0, 0.0, 0.0)
        self.web_tension = 0.0
        self.release_reason = reason
        self._previous_rope_direction = None
        self._frame_events.append(SwingEvent.RELEASE)
        if automatic:
            self.auto_releases += 1

    # ------------------------------------------------------------- physics

    def _limit_velocity(self) -> None:
        velocity = (self.vx, self.vy, self.vz)
        speed_sq = _dot(velocity, velocity)
        limit_sq = T.MAX_SPEED_TOTAL ** 2
        if speed_sq <= limit_sq:
            return

        if self.anchor is not None and self.anchor.taut:
            radius = (
                self.x - self.anchor.x,
                self.y - self.anchor.y,
                self.z - self.anchor.z,
            )
            dist = math.sqrt(_dot(radius, radius))
            if dist > 1e-6:
                outward = tuple(component / dist for component in radius)
                radial = _dot(velocity, outward)
                tangent = tuple(
                    velocity[i] - outward[i] * radial for i in range(3)
                )
                tangent_speed = math.sqrt(_dot(tangent, tangent))
                allowed = math.sqrt(max(0.0, limit_sq - radial * radial))
                if tangent_speed > allowed and tangent_speed > 1e-6:
                    scale = allowed / tangent_speed
                    velocity = tuple(
                        outward[i] * radial + tangent[i] * scale
                        for i in range(3)
                    )
                    self.vx, self.vy, self.vz = velocity
                    return

        scale = T.MAX_SPEED_TOTAL / math.sqrt(speed_sq)
        self.vx *= scale
        self.vy *= scale
        self.vz *= scale

    def _update_angular_velocity(self) -> None:
        if self.anchor is None or not self.anchor.taut:
            self.angular_velocity = (0.0, 0.0, 0.0)
            return

        radius = (
            self.x - self.anchor.x,
            self.y - self.anchor.y,
            self.z - self.anchor.z,
        )
        radius_sq = _dot(radius, radius)
        if radius_sq < 1e-8:
            self.angular_velocity = (0.0, 0.0, 0.0)
            return
        spin = _cross(radius, (self.vx, self.vy, self.vz))
        self.angular_velocity = tuple(component / radius_sq for component in spin)

    def _advance_swing_arc(self, dt: float) -> None:
        """Track cable sweep and release before a pendulum can complete a loop."""
        anchor = self.anchor
        if anchor is None:
            return
        radius = (
            self.x - anchor.x,
            self.y - anchor.y,
            self.z - anchor.z,
        )
        distance = math.sqrt(_dot(radius, radius))
        if distance < 1e-6:
            return
        direction = tuple(component / distance for component in radius)

        if self._previous_rope_direction is not None:
            sweep_axis = _cross(self._previous_rope_direction, direction)
            cross_size = math.sqrt(
                _dot(sweep_axis, sweep_axis)
            )
            dot = max(
                -1.0,
                min(1.0, _dot(self._previous_rope_direction, direction)),
            )
            self.web_arc += math.atan2(cross_size, dot)
        self._previous_rope_direction = direction
        self.web_age += dt

        upward_anchor_fraction = -direction[1]
        swept_rising_arc = self.web_arc >= math.radians(
            T.AUTO_RELEASE_SWEEP_DEG
        )
        tilted_far_enough = (
            self.web_age >= T.AUTO_RELEASE_TILT_MIN_TIME
            and upward_anchor_fraction <= math.cos(
                math.radians(T.MAX_SWING_RISE_ANGLE_DEG)
            )
        )
        reached_rising_release = (
            self.vy > 0.0
            and self.vz > 0.0
            and (swept_rising_arc or tilted_far_enough)
        )
        exceeded_arc = self.web_arc >= math.radians(T.MAX_SWING_ARC_DEG)
        if reached_rising_release or exceeded_arc:
            reason = "rise limit" if reached_rising_release else "arc limit"
            self._detach_web(reason, automatic=True)

    def _substep(self, dt: float, target_speed: float) -> None:
        restore = T.SPEED_RESTORE
        if self.anchor is not None:
            restore *= T.ATTACHED_SPEED_RESTORE

        acceleration = (
            -self.vx * T.AIR_DRAG,
            -T.GRAVITY - self.vy * T.AIR_DRAG,
            (target_speed - self.vz) * restore,
        )

        anchor = self.anchor
        old_distance = 0.0
        constraint_active = False
        if anchor is not None:
            radius = (
                self.x - anchor.x,
                self.y - anchor.y,
                self.z - anchor.z,
            )
            old_distance = math.sqrt(_dot(radius, radius))
            if anchor.taut and old_distance > 1e-6:
                outward = tuple(component / old_distance for component in radius)
                velocity = (self.vx, self.vy, self.vz)
                radial_speed = _dot(velocity, outward)
                tangent = tuple(
                    velocity[i] - outward[i] * radial_speed for i in range(3)
                )

                # omega is the actual spherical-pendulum angular velocity.
                omega = tuple(
                    component / (old_distance * old_distance)
                    for component in _cross(radius, velocity)
                )
                centripetal = _dot(omega, omega) * old_distance
                support = max(0.0, _dot(acceleration, outward) + centripetal)
                powered_pull = (
                    T.WEB_PULL_ACCEL
                    if (
                        self.web_age < T.WEB_PULL_TIME
                        and anchor.rest_length > anchor.reel_target + 1e-6
                    )
                    else 0.0
                )
                tension = support + powered_pull
                constraint_active = tension > 1e-8
                acceleration = tuple(
                    acceleration[i] - outward[i] * tension for i in range(3)
                )
                self.web_tension = max(self.web_tension, tension)

        self.vx += acceleration[0] * dt
        self.vy += acceleration[1] * dt
        self.vz += acceleration[2] * dt
        self._limit_velocity()

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        if anchor is not None:
            old_length = anchor.rest_length
            reel_floor = max(T.MIN_REST, anchor.reel_target)
            scheduled_length = max(reel_floor, old_length - T.REEL_RATE * dt)
            radius = (
                self.x - anchor.x,
                self.y - anchor.y,
                self.z - anchor.z,
            )
            predicted_distance = math.sqrt(_dot(radius, radius))

            taking_up_real_slack = (
                T.REEL_RATE > 0.0
                and old_length > reel_floor
                and predicted_distance < scheduled_length - 1e-3
            )

            if taking_up_real_slack:
                # The player is approaching faster than the base reel rate.
                # Let the powered spool follow that motion instead of applying
                # the outward impulse a passive fixed-length rope would need.
                anchor.rest_length = max(reel_floor, predicted_distance)
                anchor.taut = anchor.rest_length <= predicted_distance + 1e-7
            elif constraint_active and predicted_distance > 1e-7:
                # Positive tension means the analytical motion belongs on the
                # cable boundary. Correct both signs of tiny integration drift;
                # this is numerical cleanup, not an outward cable force.
                anchor.rest_length = scheduled_length
                outward = tuple(
                    component / predicted_distance for component in radius
                )
                correction = predicted_distance - anchor.rest_length
                self.x -= outward[0] * correction
                self.y -= outward[1] * correction
                self.z -= outward[2] * correction

                velocity = (self.vx, self.vy, self.vz)
                radial_speed = _dot(velocity, outward)
                length_rate = (anchor.rest_length - old_distance) / dt
                impulse = radial_speed - length_rate
                self.vx -= outward[0] * impulse
                self.vy -= outward[1] * impulse
                self.vz -= outward[2] * impulse
                if impulse > 0.0:
                    self.web_tension = max(self.web_tension, impulse / dt)
                anchor.taut = True
            else:
                # A powered spool may take up inward-created slack, but an
                # unpowered cable retains its length and becomes slack.
                if (
                    T.REEL_RATE > 0.0
                    and old_length > reel_floor
                    and predicted_distance < scheduled_length
                ):
                    anchor.rest_length = max(reel_floor, predicted_distance)
                else:
                    anchor.rest_length = scheduled_length

                if predicted_distance > anchor.rest_length + 1e-7:
                    outward = tuple(
                        component / predicted_distance for component in radius
                    )
                    correction = predicted_distance - anchor.rest_length
                    self.x -= outward[0] * correction
                    self.y -= outward[1] * correction
                    self.z -= outward[2] * correction

                    velocity = (self.vx, self.vy, self.vz)
                    radial_speed = _dot(velocity, outward)
                    length_rate = (anchor.rest_length - old_distance) / dt
                    if radial_speed > length_rate:
                        impulse = radial_speed - length_rate
                        self.vx -= outward[0] * impulse
                        self.vy -= outward[1] * impulse
                        self.vz -= outward[2] * impulse
                        self.web_tension = max(
                            self.web_tension, impulse / dt
                        )
                    anchor.taut = True
                elif abs(predicted_distance - anchor.rest_length) <= 1e-7:
                    anchor.taut = True
                else:
                    anchor.taut = False

        self._limit_velocity()

        if self.x < -T.LATERAL_LIMIT:
            self.x = -T.LATERAL_LIMIT
            self.vx = max(0.0, self.vx)
        elif self.x > T.LATERAL_LIMIT:
            self.x = T.LATERAL_LIMIT
            self.vx = min(0.0, self.vx)

        if self.y > T.CEILING_Y:
            self.y = T.CEILING_Y
            self.vy = min(0.0, self.vy)

        self._update_angular_velocity()
        self._advance_swing_arc(dt)

        if self.y <= T.STREET_DEATH_Y:
            self.y = T.STREET_DEATH_Y
            self.alive = False
            self.death_reason = "hit the street"

    # --------------------------------------------------------------- update

    def update(
        self, dt: float, ctrl: ControlState, world: WorldStrip
    ) -> tuple[SwingEvent, ...]:
        self._frame_events = []
        self.last_events = ()
        if not self.alive or dt <= 0.0:
            return self.last_events

        self.elapsed += dt
        target_speed = min(T.MAX_SPEED, T.START_SPEED + self.elapsed * T.SPEED_RAMP)

        # One web per pose. Releasing preserves the instantaneous tangential
        # launch velocity. A rising-angle assist and a swept-arc safety cap may
        # also release it before the player can orbit the anchor.
        if not ctrl.thwip_held:
            if self.anchor is not None:
                self._detach_web("player release")
            self._armed = True
        elif self.anchor is None and self._armed:
            self.shots += 1
            self._frame_events.append(SwingEvent.SHOT)
            found = self._pick_anchor(world, ctrl)
            self._armed = False
            if found is None:
                self.whiffs += 1
                self._frame_events.append(SwingEvent.MISS)
            else:
                self.anchor = found
                self._catch_web(found)
                self.attaches += 1
                self._frame_events.append(SwingEvent.ATTACH)

        self.web_tension = 0.0
        steps = max(1, math.ceil(dt / T.PHYSICS_MAX_STEP))
        step_dt = dt / steps
        for _ in range(steps):
            self._substep(step_dt, target_speed)
            if not self.alive:
                break

        self.peak_y = max(self.peak_y, self.y)
        self.low_y = min(self.low_y, self.y)
        self.last_events = tuple(self._frame_events)
        return self.last_events

    # ---------------------------------------------------------------- state

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)

    @property
    def angular_speed(self) -> float:
        return math.sqrt(_dot(self.angular_velocity, self.angular_velocity))

    @property
    def attached(self) -> bool:
        return self.anchor is not None

    @property
    def rope_taut(self) -> bool:
        return self.anchor is not None and self.anchor.taut
