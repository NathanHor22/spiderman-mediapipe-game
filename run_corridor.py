"""Step 2 — corridor renderer.

Flies the camera down an endless procedural street canyon on a canned path.
There is no input model and no physics here on purpose: this exists to answer
one question, which is whether the look is right, before any of the swing
feel depends on it.

The lazy sine sway is a stand-in for a swing arc — it is roughly the amplitude
and period a real pendulum will produce, so what you see here is close to what
the game will feel like to look at.

  W / S    faster / slower
  A / D    drift left / right
  R / F    climb / descend
  X (hold) preview spider-sense slow motion
  TAB      toggle windows
  SPACE    toggle the automatic sway
  ESC      quit
"""

from __future__ import annotations

import math

import pygame

from spidergame.clock import GameClock
from spidergame.render.projection import Camera3D
from spidergame.render.renderer import Renderer
from spidergame.render.world import WorldStrip

WIDTH, HEIGHT = 1280, 720
BASE_SPEED = 78.0  # world units per second


def main() -> int:
    pygame.init()
    pygame.display.set_caption("spider corridor - renderer preview")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas", 16)

    renderer = Renderer(WIDTH, HEIGHT)
    world = WorldStrip(seed=11)
    cam = Camera3D(x=0.0, y=34.0, z=0.0)
    clock_src = pygame.time.Clock()
    clock = GameClock()

    speed = BASE_SPEED
    sway = True
    show_windows = True
    show_stats = True
    manual_x = 0.0
    manual_y = 0.0
    prev_x = 0.0

    running = True
    while running:
        clock.tick(clock_src.tick(60) / 1000.0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    show_windows = not show_windows
                elif event.key == pygame.K_SPACE:
                    sway = not sway
                elif event.key == pygame.K_F1:
                    show_stats = not show_stats

        keys = pygame.key.get_pressed()
        rdt = clock.real_dt

        # Slow-mo is driven off real time so the ramp feels identical no matter
        # how dilated the world already is.
        if keys[pygame.K_x]:
            clock.enter_slowmo(0.25)
        else:
            clock.exit_slowmo()

        if keys[pygame.K_w]:
            speed += 60.0 * rdt
        if keys[pygame.K_s]:
            speed -= 60.0 * rdt
        speed = max(0.0, min(speed, 260.0))

        if keys[pygame.K_a]:
            manual_x -= 14.0 * rdt
        if keys[pygame.K_d]:
            manual_x += 14.0 * rdt
        if keys[pygame.K_r]:
            manual_y += 20.0 * rdt
        if keys[pygame.K_f]:
            manual_y -= 20.0 * rdt

        dt = clock.game_dt
        cam.z += speed * dt

        t = clock.game_time
        if sway:
            swing_x = math.sin(t * 0.85) * 7.5
            swing_y = math.sin(t * 1.70 + 0.4) * 6.5
        else:
            swing_x = swing_y = 0.0

        cam.x = swing_x + manual_x
        cam.y = max(4.0, 34.0 + swing_y + manual_y)

        # Bank into the drift. Costs nothing and does more for the sense of
        # speed than any amount of extra geometry.
        lateral_v = (cam.x - prev_x) / dt if dt > 1e-6 else 0.0
        prev_x = cam.x
        cam.roll = max(-0.09, min(0.09, -lateral_v * 0.004))

        world.update(cam.z)
        stats = renderer.render(screen, cam, world, show_windows=show_windows)

        if show_stats:
            lines = [
                f"fps {clock_src.get_fps():5.1f}   faces {stats['faces']:4d}   windows {stats['windows']:4d}",
                f"speed {speed:6.1f}   z {cam.z:8.1f}   y {cam.y:5.1f}   x {cam.x:6.2f}",
                f"time_scale {clock.time_scale:4.2f}   buildings {len(world.buildings):3d}",
            ]
            for i, line in enumerate(lines):
                screen.blit(font.render(line, True, (200, 200, 210)), (12, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
