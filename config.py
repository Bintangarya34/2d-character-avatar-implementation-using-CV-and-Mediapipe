"""
Configuration file - Semua setting & konstanta di sini
"""

# --- ASSET PATHS ---
assets = {
    'body': 'body_front.png',
    'body_left': 'body_left.png',
    'body_right': 'body_right.png',
    'head': 'head.png',
    'head_left': 'head_left.png',
    'head_right': 'head_right.png',
    'right_arm_upper': 'arm_upper.png',
    'right_arm_lower': 'arm_lower.png',
    'left_arm_upper': 'arm_upper.png',
    'left_arm_lower': 'arm_lower.png',
    'right_leg_upper': 'upper_leg_right.png',
    'right_leg_lower': 'lower_leg_right.png',
    'left_leg_upper': 'upper_leg_left.png',
    'left_leg_lower': 'lower_leg_left.png',
}

# --- DISPLAY ---
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_NAME = 'VTuber Avatar'
FPS_TARGET = 60
FRAME_TIME_MS = 16  # 1000 / 60

# --- BACKGROUND ---
BG_COLOR = [255, 255, 255]  # Putih (BGR)

# --- MEDIAPIPE SETTINGS ---
POSE_MIN_DETECTION_CONFIDENCE = 0.2
POSE_MIN_TRACKING_CONFIDENCE = 0.2
POSE_MODEL_COMPLEXITY = 0  # Lite mode

HANDS_MIN_DETECTION_CONFIDENCE = 0.2
HANDS_MODEL_COMPLEXITY = 0  # Lite mode
HANDS_MAX_NUM = 2

# --- BODY SCALING ---
SHOULDER_DISTANCE_REFERENCE = 200.0  # Reference distance untuk scale
BODY_SCALE_MIN = 0.5
BODY_SCALE_MAX = 2.0

# --- HEAD SCALING ---
HEAD_SCALE_MULTIPLIER = 1.5
HEAD_POSITION_Y_OFFSET = 0.85  # Large offset agar kepala FULLY di atas shoulder
EYE_SCALE_MULTIPLIER = 15

# --- BODY PARTS SCALING ---
BODY_SCALE_WIDTH = 1.5
ARM_SCALE_WIDTH = 1.5

# --- HAND ---
HAND_WRIST_RADIUS_MULTIPLIER = 12
HAND_PALM_THICKNESS_MULTIPLIER = 3
FINGER_THICKNESS_MULTIPLIER = 4
FINGER_TIP_RADIUS_MULTIPLIER = 1.2

# --- BRUSH ---
BRUSH_HANDLE_LENGTH = 300
BRUSH_HEAD_LENGTH = 80
BRUSH_WIDTH = 20
BRUSH_COLOR = (200, 150, 100)  # Cokelat muda (BGR)
BRUSH_BRISTLE_COLOR = (100, 100, 255)  # Merah muda (BGR)
BRUSH_MIN_GRAB_DISTANCE = 80  # Pixel max untuk grab detection (saat tangan dekat/genggam)

# --- DRAWING COLORS (Random pada setiap grab berakhir) ---
DRAWING_COLORS = [
    (0, 0, 255),      # Merah
    (0, 165, 255),    # Orange
    (0, 255, 255),    # Kuning
    (0, 255, 0),      # Hijau
    (255, 0, 0),      # Biru
    (255, 0, 255),    # Magenta
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
]

# --- DRAWING CANVAS ---
DRAWING_BLEND_RATIO = 0.2  # 20% drawing opacity

# --- SMOOTHING ---
LANDMARK_SMOOTHING_FACTOR = 0.7  # EMA factor (0.0-1.0, higher = smoother)

# --- ANIMATION ---
BLINK_CHANCE_PER_FRAME = 0.02  # 2% per frame
BLINK_DURATION_MIN = 1
BLINK_DURATION_MAX = 3

# --- CAMERA ---
CAMERA_DEVICE = 0  # Default camera
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
CAMERA_FPS = 60
CAMERA_BUFFER_SIZE = 1  # Minimal buffer untuk low latency

# --- ASSET OPTIMIZATION ---
ASSET_MAX_HEIGHT = 600
ASSET_RESIZE_INTERPOLATION = 'INTER_CUBIC'
