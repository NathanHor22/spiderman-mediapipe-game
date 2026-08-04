"""Every number that decides how swinging feels, in one place.

Values are in world units (roughly metres) and seconds.
"""

# --- forward motion ---------------------------------------------------------
# Forward speed is not player-controlled, as in Subway Surfers. Free flight
# restores it strongly; the attached multiplier below gets out of the way of
# pendulum momentum while the web is taut.
START_SPEED = 52.0
MAX_SPEED = 128.0
SPEED_RAMP = 0.55
SPEED_RESTORE = 1.7
ATTACHED_SPEED_RESTORE = 0.20

# --- gravity ----------------------------------------------------------------
# Stronger than earth gravity, but low enough for a taut 25-unit web to complete
# a readable quarter-arc in about one second.
GRAVITY = 65.0

# --- the web ----------------------------------------------------------------
# An inextensible one-sided constraint, solved in short substeps. It pulls when
# taut, permits inward slack, and remains stable through a 50ms game-clock hitch.
PHYSICS_MAX_STEP = 1.0 / 240.0
REEL_RATE = 12.0
MIN_REST = 12.0
MAX_WEB_RANGE = 95.0

# The web shooter is powered rather than passive. Attachment gives one bounded
# impulse along the high cable, then a mild central pull and reel take over.
# Central pull adds no angular torque; gravity still shapes the pendulum arc.
WEB_CATCH_UP_SPEED = 5.0
WEB_CATCH_MAX_DV = 28.0
WEB_PULL_ACCEL = 4.0

# --- anchor placement -------------------------------------------------------
# Hand x picks the side/reach and hand y picks height. A minimum upward cable
# direction prevents accurate tension from becoming a mostly-forward boost.
ANCHOR_AHEAD_MIN = 7.0
ANCHOR_AHEAD_MAX = 18.0
ANCHOR_HEIGHT_MIN = 64.0
ANCHOR_HEIGHT_MAX = 116.0
ANCHOR_MIN_CLEARANCE = 6.0
ANCHOR_MIN_RISE = 10.0
MIN_ANCHOR_UP_DOT = 0.45

# Preserves the player's raised starting altitude while high-anchor clamping
# keeps every successful web shot useful from that height.
START_Y = 50.0

# --- bounds -----------------------------------------------------------------
STREET_DEATH_Y = 3.0
CEILING_Y = 140.0
LATERAL_LIMIT = 6.5
MAX_SPEED_TOTAL = 190.0

# --- feel -------------------------------------------------------------------
AIR_DRAG = 0.05
CAMERA_LAG = 7.0
CAMERA_BACK = 24.0
CAMERA_UP = 2.5
CAMERA_ROLL_GAIN = 0.0030
CAMERA_ROLL_MAX = 0.10
