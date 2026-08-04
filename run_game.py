"""The game.

Keyboard by default; `--vision` uses the webcam. Because everything below
ControlState is input-agnostic, that flag is the only difference between them —
the physics, the tutorial and the HUD have no idea which is running.

  --vision          use the webcam instead of keyboard + mouse
  --camera N        start on capture index N (switchable in-game with [ and ])
  --list-cameras    probe the machine's cameras and exit
  --skip-tutorial   go straight to the countdown

  hold SPACE / the web-shooter sign   thwip. One web per press: you must
                                      release before firing another.
  mouse / hand position               which side you anchor to, and how high
  X (hold)                            preview spider-sense slow motion
  R  restart      TAB windows      F1 stats      ESC back / quit
"""

from __future__ import annotations

import argparse
import math
import sys

import pygame

from spidergame.audio import SoundSystem, prepare_mixer
from spidergame.clock import GameClock
from spidergame.control import ControlState
from spidergame.game import tuning as T
from spidergame.game.swing import SwingSim
from spidergame.producers import KeyboardProducer
from spidergame.render.actor import draw_web, player_faces
from spidergame.render.projection import Camera3D
from spidergame.render.renderer import Renderer
from spidergame.render.world import WorldStrip
from spidergame.ui import bgr_to_surface, hud, screens

WIDTH, HEIGHT = 1280, 720

TITLE, TUTORIAL, COUNTDOWN, PLAYING, DEAD = range(5)
COUNTDOWN_STEPS = ("3", "2", "1", "GO")
COUNTDOWN_STEP_S = 0.55


def open_vision(index: int, font, screen, timeout: float = 45.0):
    """Start the vision thread, drawing a status frame first since it blocks."""
    from spidergame.producers import VisionProducer

    screen.fill((8, 7, 14))
    hud.draw_text(screen, font, f"connecting to camera {index}...", (40, 40), hud.INK)
    hud.draw_text(screen, font, "(first run downloads the hand model)",
                  (40, 66), hud.DIM)
    pygame.display.flip()
    pygame.event.pump()

    producer = VisionProducer(camera_index=index, keep_frame=True)
    if not producer.wait_ready(timeout):
        err = producer.error or "timed out"
        producer.close()
        return None, err
    return producer, None


def _try_camera_switch(current, current_index: int, new_index: int, opener):
    """Replace a live camera only after its candidate is confirmed ready."""
    candidate, error = opener(new_index)
    if candidate is None:
        return (
            current,
            current_index,
            f"camera {new_index}: {error}",
            False,
        )

    current.close()
    return candidate, new_index, None, True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--camera", type=int, default=None,
                    help="capture index; omit to auto-pick a working camera")
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--skip-tutorial", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    if args.list_cameras:
        from spidergame.vision import devices

        print("probing capture indices 0-3...")
        for info in devices.available():
            flag = "   <-- BLACK FRAMES" if info.dark else ""
            print(f"  {info.label()}  brightness {info.brightness:5.1f}{flag}")
        print("\nwindows device list (order does NOT match capture indices):")
        for name in devices.system_names():
            print(f"  {name}")
        return 0

    prepare_mixer()
    pygame.init()
    pygame.display.set_caption("spider — endless swinger")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont("consolas", 16)
    mid = pygame.font.SysFont("consolas", 22)
    big = pygame.font.SysFont("consolas", 46, bold=True)
    fonts = (font, mid, big)
    clock_src = pygame.time.Clock()
    audio = SoundSystem()

    camera_index = args.camera if args.camera is not None else 0
    producer = None
    vision_error = None
    auto_picked = False

    if args.vision:
        from spidergame.vision import devices

        if args.camera is None:
            screen.fill((8, 7, 14))
            hud.draw_text(screen, font, "looking for a working camera...",
                          (40, 40), hud.INK)
            pygame.display.flip()
            pygame.event.pump()
            found = devices.pick_default()
            if found is None:
                print("no camera delivering usable frames — check the privacy "
                      "shutter and Windows camera permissions", file=sys.stderr)
                audio.close()
                pygame.quit()
                return 1
            camera_index = found
            auto_picked = True

        producer, vision_error = open_vision(camera_index, font, screen)
        if producer is None:
            print(f"vision failed: {vision_error}", file=sys.stderr)
            audio.close()
            pygame.quit()
            return 1
    else:
        producer = KeyboardProducer()

    renderer = Renderer(WIDTH, HEIGHT)
    clock = GameClock()

    # The menus sit over a live corridor on its own world, so the title screen
    # shows the game moving instead of a flat colour.
    backdrop_world = WorldStrip(seed=args.seed + 7)
    backdrop_cam = Camera3D(x=0.0, y=36.0, z=0.0)

    world = WorldStrip(seed=args.seed)
    sim = SwingSim()
    cam = Camera3D(x=0.0, y=sim.y + T.CAMERA_UP, z=sim.z - T.CAMERA_BACK)

    state = COUNTDOWN if args.skip_tutorial else TITLE
    title_menu = screens.TitleMenu()
    title_menu_rects: tuple[pygame.Rect, ...] = ()
    flow = None
    countdown_t = 0.0
    show_windows = True
    show_stats = False
    best_distance = 0.0
    pulse_t = 0.0

    def reset_run():
        nonlocal world, sim
        audio.stop()
        world = WorldStrip(seed=args.seed)
        sim = SwingSim()
        cam.x, cam.y, cam.z = 0.0, sim.y + T.CAMERA_UP, sim.z - T.CAMERA_BACK
        cam.roll = 0.0

    def switch_camera(delta: int):
        nonlocal producer, camera_index, vision_error, auto_picked
        if not args.vision:
            return
        new_index = max(0, camera_index + delta)
        if new_index == camera_index:
            return
        producer, camera_index, vision_error, switched = _try_camera_switch(
            producer,
            camera_index,
            new_index,
            lambda index: open_vision(index, font, screen, timeout=10.0),
        )
        if switched:
            auto_picked = False

    running = True

    def activate_title(action: str | None) -> bool:
        """Apply one title action and report whether it consumed the input."""
        nonlocal state, countdown_t, flow, running
        if action == "start":
            reset_run()
            state, countdown_t = COUNTDOWN, 0.0
        elif action == "training":
            flow = screens.TutorialFlow(args.vision)
            state = TUTORIAL
        elif action == "quit":
            running = False
        else:
            return False
        return True

    while running:
        clock.tick(clock_src.tick(60) / 1000.0)
        rdt = clock.real_dt
        pulse_t += rdt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEMOTION and state == TITLE:
                title_menu.select_at(event.pos, title_menu_rects)
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and state == TITLE
            ):
                activate_title(
                    title_menu.activate_at(event.pos, title_menu_rects)
                )
            elif event.type == pygame.KEYDOWN:
                title_action = (
                    title_menu.handle_event(event) if state == TITLE else None
                )
                if activate_title(title_action):
                    pass
                elif event.key == pygame.K_ESCAPE:
                    if state in (PLAYING, DEAD, TUTORIAL, COUNTDOWN):
                        audio.stop()
                        state = TITLE
                    else:
                        running = False
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    if state == DEAD:
                        state = TITLE
                elif event.key == pygame.K_t and state == DEAD:
                    flow = screens.TutorialFlow(args.vision)
                    state = TUTORIAL
                elif event.key == pygame.K_r and state in (PLAYING, DEAD):
                    reset_run()
                    state = PLAYING
                elif event.key == pygame.K_TAB:
                    show_windows = not show_windows
                elif event.key == pygame.K_F1:
                    show_stats = not show_stats
                elif event.key == pygame.K_LEFTBRACKET and state == TITLE:
                    switch_camera(-1)
                elif event.key == pygame.K_RIGHTBRACKET and state == TITLE:
                    switch_camera(1)
            if isinstance(producer, KeyboardProducer):
                producer.handle_event(event)

        # Do not poll input, advance physics, or start a last sound after the
        # window or title menu has already requested shutdown.
        if not running:
            break

        ctrl = producer.poll()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_x] and state == PLAYING:
            clock.enter_slowmo(0.25)
        else:
            clock.exit_slowmo()

        cam_surface = None
        if args.vision and state in (TITLE, TUTORIAL):
            frame, _ = producer.worker.debug_frame()
            if frame is not None:
                cam_surface = bgr_to_surface(frame)

        # Commit flow transitions before choosing which world to render. This
        # prevents a countdown expiry from drawing gameplay HUD over the menu
        # backdrop, and makes tutorial completion show "3" immediately.
        if state == TUTORIAL:
            flow.update(ctrl, rdt)
            if flow.done:
                reset_run()
                state, countdown_t = COUNTDOWN, 0.0
        elif state == COUNTDOWN:
            countdown_t += rdt
            if int(countdown_t / COUNTDOWN_STEP_S) >= len(COUNTDOWN_STEPS):
                state = PLAYING

        # ---------------------------------------------------------- update
        if state in (TITLE, TUTORIAL, COUNTDOWN):
            backdrop_cam.z += 46.0 * rdt
            backdrop_cam.x = math.sin(pulse_t * 0.5) * 5.0
            backdrop_cam.y = 36.0 + math.sin(pulse_t * 0.8) * 4.0
            backdrop_world.update(backdrop_cam.z)
            renderer.render(screen, backdrop_cam, backdrop_world,
                            show_windows=show_windows)
        else:
            dt = clock.game_dt
            if state == PLAYING:
                world.update(sim.z + 60.0)
                swing_events = sim.update(dt, ctrl, world)
                audio.handle(swing_events, sim)
                best_distance = max(best_distance, sim.z)
                if not sim.alive:
                    audio.stop()
                    state = DEAD

            lag = min(1.0, T.CAMERA_LAG * dt) if dt > 0 else 0.0
            cam.x += (sim.x - cam.x) * lag
            cam.y += ((sim.y + T.CAMERA_UP) - cam.y) * lag
            cam.z = sim.z - T.CAMERA_BACK
            cam.roll = max(-T.CAMERA_ROLL_MAX,
                           min(T.CAMERA_ROLL_MAX, -sim.vx * T.CAMERA_ROLL_GAIN))
            speed_t = min(max((sim.vz - T.START_SPEED)
                              / (T.MAX_SPEED - T.START_SPEED), 0.0), 1.0)
            cam.fov_deg = 76.0 + 12.0 * speed_t

            renderer.render(screen, cam, world, show_windows=show_windows,
                            actors=[(player_faces(sim), sim.z)])
            draw_web(screen, sim, cam, WIDTH * 0.5, HEIGHT * 0.5, cam.focal(WIDTH),
                     math.cos(cam.roll), math.sin(cam.roll))

        # ------------------------------------------------------------ draw
        pulse = math.sin(pulse_t * 3.0)

        if state == TITLE:
            info = "[ / ] switch camera" if args.vision else ""
            if auto_picked:
                info += "  |  auto-selected"
            if vision_error:
                info = f"({vision_error})  |  [ / ] switch camera"
            title_menu_rects = screens.draw_title(
                screen,
                fonts,
                vision=args.vision,
                camera_index=camera_index,
                camera_info=info,
                pulse=pulse,
                cam_surface=cam_surface,
                ctrl=ctrl,
                selected=title_menu,
                sound_available=audio.enabled,
            )

        elif state == TUTORIAL:
            screens.draw_tutorial(screen, fonts, flow, ctrl, cam_surface,
                                  pulse)

        elif state == COUNTDOWN:
            idx = int(countdown_t / COUNTDOWN_STEP_S)
            screens.draw_countdown(screen, fonts, COUNTDOWN_STEPS[idx])

        if state in (PLAYING, DEAD):
            hud.draw_text(screen, big, f"{sim.z:6.0f} m", (24, 20), hud.INK)
            alt_colour = (hud.BAD if sim.y < 12 else
                          hud.WARN if sim.y < 20 else hud.GOOD)
            hud.draw_text(screen, font, f"altitude {sim.y:5.1f}", (26, 78),
                          alt_colour)
            hud.draw_text(screen, font,
                          "WEB ATTACHED" if sim.attached else "falling",
                          (26, 98), hud.GOOD if sim.attached else hud.DIM)
            hud.draw_bar(screen, pygame.Rect(26, 120, 180, 12),
                         max(0.0, sim.y - T.STREET_DEATH_Y), 70.0, alt_colour)
            if ctrl.tracking_lost:
                hud.draw_text(screen, font, "NO HAND — raise your hand",
                              (26, 142), hud.BAD)

            if show_stats:
                lines = [
                    f"fps {clock_src.get_fps():5.1f}",
                    f"vel ({sim.vx:6.1f},{sim.vy:6.1f},{sim.vz:6.1f})",
                    f"omega {sim.angular_speed:5.2f} rad/s  "
                    f"tension {sim.web_tension:6.1f}  "
                    f"rope {'taut' if sim.rope_taut else 'slack'}",
                    f"attach {sim.attaches:3d}  whiff {sim.whiffs:3d}",
                    f"{sim.whiff_reason}",
                    f"time_scale {clock.time_scale:4.2f}  "
                    f"hand ({ctrl.hand_x:.2f},{ctrl.hand_y:.2f})",
                ]
                for i, line in enumerate(lines):
                    hud.draw_text(screen, font, line,
                                  (WIDTH - 470, 16 + i * 19), hud.DIM)

        if state == DEAD:
            box = pygame.Rect(WIDTH // 2 - 270, HEIGHT // 2 - 90, 540, 180)
            screens.panel(screen, box, alpha=232, edge=hud.BAD)
            hud.draw_text(screen, big, sim.death_reason.upper(),
                          (box.x + 30, box.y + 26), hud.BAD)
            hud.draw_text(screen, mid, f"{sim.z:.0f} m", (box.x + 30, box.y + 88),
                          hud.INK)
            hud.draw_text(screen, font, f"best {best_distance:.0f} m",
                          (box.x + 30, box.y + 120), hud.DIM)
            hud.draw_text(screen, font, "R restart     ENTER title     T tutorial",
                          (box.x + 30, box.y + 144), hud.DIM)

        pygame.display.flip()

    try:
        producer.close()
    finally:
        try:
            audio.close()
        finally:
            pygame.quit()
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        # Idempotent, and still tears SDL down if an unexpected frame-loop
        # exception bypasses main's normal producer/audio cleanup path.
        pygame.quit()
    raise SystemExit(exit_code)
