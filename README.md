# Spider-Man — gesture-controlled endless swinger

Subway Surfers structure, PS1 Spider-Man look, webcam for a controller.

You fly down an endless street canyon. Holding the **thwip** sign attaches a web
to a building on whichever side your hand is on, and the pendulum swing that
follows is both your lateral movement and your jump. Stop thwipping and gravity
takes you to street level. Green Goblin turns up on a glider and throws pumpkin
bombs; each one drops the world into spider-sense slow motion and you have to
**punch** it out of the air.

## Design in one paragraph

Hand tracking is a discrete, low-precision, high-latency input, so the game is
built around what it is actually good at rather than fighting it. `thwip` is a
*sustained* pose, which MediaPipe is far steadier at than transients, and its
latency reads as web travel time rather than input lag. `punch` is a transient
and would normally be the weak point — except it only ever matters inside
bullet-time, where a 200ms detection pipeline costs 50ms of game time. The
slow-motion is what makes the unreliable gesture reliable, and it happens to be
literally what spider-sense is.

## Setup

```
pip install -r requirements.txt
```

The hand landmarker model (~8MB) downloads automatically on first run.

## What runs today

Everything is pygame, including the diagnostic tools. `cv2.imshow` has no sound,
no real event handling and its own frame pacing, so tools built on it drift away
from how the game actually behaves — which defeats the point of a tool you use
to judge whether the game feels right. OpenCV is now only used for camera
capture and colour conversion.

**`python run_game.py`** — the game. Keyboard by default; `--vision` swaps in the
webcam and nothing else changes.

| key | |
|---|---|
| `↑` / `↓` or `W` / `S`, then `ENTER` | choose a title-screen action |
| mouse hover + click | select and activate a title-screen action |
| hold `SPACE` / left mouse | thwip |
| mouse position | which side you anchor to, and how high |
| `X` hold | preview spider-sense slow motion |
| `R` | restart |
| `TAB` / `F1` | windows / stats |

**One web per thwip.** You must release before firing another, so choosing when
to let go *is* the game. Without that gate, holding the sign down forever was the
dominant strategy because a missed shot could retry on every frame. A successful
web reels in by a small, fixed amount and pulls upward, then either releases when
you relax the pose or assists the release near the top of the rising arc. A swept-
arc safety cap prevents unusual approaches from turning into endless loops.

The comic-style starting screen has START, TRAINING and QUIT choices over a live
corridor, plus an aspect-correct camera preview and live hand/gesture status in
vision mode. START goes directly to the countdown; TRAINING opens the interactive
lesson. The tutorial gates on you actually performing each action — hold the
sign, let go, then sweep left and right — rather than on a timer. Losing hand
tracking deliberately does not satisfy the "let go" step, or you could skip it
by dropping your hand out of frame. `--skip-tutorial` bypasses the title as well
and starts at the countdown.

Web shots, catches, misses and releases use procedural sound effects, so there
are no external audio assets to install. While attached, a looping stereo whoosh
tracks swing speed, angular velocity, tension and cable slack. If SDL cannot open
an audio device, the game continues silently instead of failing to start.

### Camera selection

Capture indices are not stable. Plugging in a USB camera reshuffles them, and a
laptop's built-in camera can sit at index 0 returning black frames behind a
privacy shutter — on this machine index 0 went from 30fps/bright to
1fps/black between two runs for exactly that reason.

So `--vision` **auto-picks the first index that delivers real pixels**, and `[`
and `]` cycle cameras live on the title screen with a preview inset. Pick the one
you can see yourself in.

```
python run_game.py --list-cameras     # probe indices + Windows device names
python run_game.py --vision           # auto-pick
python run_game.py --vision --camera 1
```

`--list-cameras` prints Windows device names for orientation and annotates a
built-in vs external guess from the USB vendor ID, but that ordering does **not**
map to OpenCV's capture indices. The title therefore uses the live preview and
capture index rather than displaying a misleading name beside it.


**`python run_gesture_harness.py`** — the tuning rig. Webcam feed, landmark
overlay, and a very loud readout of what the classifier believes. Run this
first.

The web-shooter pose is **index + pinky extended, middle + ring curled**. The
thumb may be open or tucked, and either a palm-facing or back-of-hand view is
accepted. Detection uses continuous 3D finger-straightness confidence: forming
the pose has a strict threshold, while an already-held pose gets a relaxed
threshold so a briefly hidden curled fingertip does not drop the web. Palm aim is
smoothed with a time-based 80ms filter to remove landmark jitter without adding a
frame-rate dependency.

| key | |
|---|---|
| `C` | calibrate punch threshold (throw three punches) |
| `[` `]` | nudge the threshold by hand |
| `S` | save calibration |
| `L` | toggle landmarks |
| `ESC` | quit |

Two things to watch:

- **`vision fps`, top right.** This is the number that decides how good
  everything else can feel. If it is under ~20, drop `width`/`height` in
  `VisionWorker` to 480×360, or `num_hands` to 1.
- **The growth-rate bar.** Move around normally and watch how close the bar gets
  to the threshold marker. You want obvious headroom — a threshold that only
  just clears your idle motion will fire on its own mid-game.

Run the geometry, timing and swing-height regression checks with:

```
python -m unittest discover -v
```

**`python run_corridor.py`** — the renderer preview. Flies a canned path down the
canyon with a sine sway standing in for a swing arc. No physics; this is here to
judge the look. `W`/`S` speed, `A`/`D` drift, `R`/`F` altitude, `X` preview
slow-motion, `TAB` windows, `SPACE` sway, `F1` stats.

Around 100fps headless at 1280×720 with ~100 building faces and ~100 lit
windows, so there is real headroom for physics and entities.

## Layout

```
spidergame/
  audio.py              procedural web, cable and swing effects
  control.py            ControlState — the seam between input and game
  clock.py              real_dt vs game_dt (time dilation)
  vision/
    gestures.py         finger geometry, thwip/fist, punch detector, hysteresis
    worker.py           capture thread + inference thread + classification
    calibration.py      per-player punch threshold
    models.py           downloads hand_landmarker.task
  producers/
    keyboard.py         tuning rig input
    vision.py           webcam input
  game/
    tuning.py           every number that decides how swinging feels
    swing.py            rope physics, anchor selection, fail state
  render/
    projection.py       perspective, near/segment clipping, back-face culling
    world.py            procedural canyon, generates ahead, retires behind
    renderer.py         painter's algorithm, flat shading, fog
    actor.py            player figure and web line
    palette.py          colours, fog, and the colour-codes-the-verb rule
  ui/
    surface.py          camera frames -> pygame surfaces
    hud.py              text, meters, panels
```

### Two rules worth not breaking

**Everything downstream of `ControlState` is input-agnostic.** Physics gets
tuned on `KeyboardProducer`, where a bad number is unambiguously a bad number
rather than a dropped landmark frame. Swapping in `VisionProducer` afterwards is
a one-line change.

**Physics reads `game_dt`, vision reads `real_dt`.** Nothing about the camera or
the gesture pipeline may ever be slowed by time dilation — running detection at
full speed while the world crawls is the whole point.

## Build order

- [x] 1. Gesture harness — thwip + punch, with calibration
- [x] 2. Corridor renderer
- [x] 3. Swing physics, on the keyboard producer, with scaled dt from the start
- [x] 4. Vision producer swap (`--vision`) — plumbed, not yet played with a hand
- [ ] 5. Static obstacles, collision, fall-to-street death
- [ ] 6. Goblin + bomb QTE on the time-scale system
- [ ] 7. Scoring, speed ramp, juice

### Swing model

The web is a substepped, one-sided distance constraint rather than a Hooke
spring. A taut line supplies the exact radial support and centripetal tension;
gravity's remaining tangential component changes the 3D angular velocity. A
slack line never pushes the player.

Attachment starts at the real player-to-anchor distance, then a bounded powered
catch and six-unit reel pull the character upward without teleporting them.
Anchors are clamped to a useful elevation and the buildings span 72-132 units so
the raised 50-unit start still has reliable attachment choices. If the nearest
roof is too low, the shot checks a taller neighbouring building before becoming
a miss. Forward speed is not artificially restored while attached, gravity
changes the angular velocity, and a forward/upward 55-degree sweep or rope-tilt
release prevents a full orbit while preserving instantaneous launch velocity. A
larger swept-arc cap remains as a numerical safety fallback.

## Open issue: bimodal inference speed

On this machine `detect_for_video` lands in one of two states and stays there:

| state | inference | pipeline | dropped frames |
|---|---|---|---|
| good | ~15 ms | ~30 fps | 0 |
| bad | ~100 ms | ~9 fps | ~110 / 10s |

Which one you get appears random per process. Ruled out by measurement, none of
these changed it:

- `num_hands` 1 vs 2 — no difference (an early 2x reading, and a later 8x
  reading in the other direction, were both noise)
- capture resolution, and downscaling before inference — no difference, because
  MediaPipe rescales internally anyway
- IMAGE vs VIDEO running mode
- counter timestamps vs real elapsed-ms timestamps (no throughput difference;
  real timestamps are still used for correct video tracking)
- main thread vs background thread, with and without a parallel capture loop
- `cv2.flip`, the colour convert, frame copies, and the gesture classifier —
  all together are under 2ms and none of them move the number

Everything in the pipeline outside `detect_for_video` is negligible. The CPU
itself benchmarks healthy (95 GFLOPS, normal single-thread). GPU delegate is not
an option: the Windows wheel ships with `GPU processing is disabled in build
flags`.

That leaves something outside the process — power management, background CPU
contention, or thermal. The machine is on the **Balanced** power plan, which is
the first thing to try changing. The harness reports capture fps, inference ms
and dropped frames separately so you can see which state you are in at a glance.

At ~15ms the design works as intended. At ~100ms gesture acquisition is still
limited by how quickly a new inference result arrives, but the latch itself is
time-based so release/re-arm does not stretch to five slow frames. The punch QTE
still works because bullet-time is doing the heavy lifting.

### What screenshots turned up that tests did not

Rendering "without exceptions" is not the same as rendering correctly. Three
defects were only visible by actually looking at a frame:

- **Windows drew with no wall behind them.** They skipped back-face culling on
  the assumption their wall was visible — but a building being inside the z
  range does not mean its inner face still faces you. Once the camera drew level
  with a facade, its windows carried on painting as lit rectangles floating in
  mid-air.
- **The player was cropped off the bottom edge.** At `CAMERA_UP 5.5` / `BACK 19`
  the figure projected ~240px below centre. Now 2.5 / 24.
- **`LATERAL_LIMIT` let the player hug a wall.** At 12 against a canyon
  half-width of 14 you could sit two units off a facade, filling half the screen
  with blank concrete. Now 6.5.

Wide gaps between buildings also exposed raw sky, which read as a hole rather
than an alley, so the rows are now near-continuous (gaps 0.5-3.5, footprints
20-40).

## Notes and known limits

- The sky is a precomputed gradient with the horizon pinned to the screen
  midpoint, so it does not bank with camera roll. Roll is kept under ~5°, where
  the mismatch is invisible under fog. Adding camera *pitch* would break this and
  means making the sky real geometry.
- `WorldStrip.nearest_ahead()` exists and is unused — it is the hook the web
  anchor system will attach to in step 3.
- A missed punch should knock the player off their web with ~1.5s to recover,
  not kill outright. Death by webcam misread feels unfair even when it is the
  player's fault; routing it back through the fall-to-street fail state keeps
  every death traceable to "I didn't get my web out in time".
