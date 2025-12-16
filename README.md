# VTuber Avatar with Pose & Hand Tracking

Aplikasi real-time VTuber avatar yang mendeteksi pose tubuh dan tangan user, kemudian render character yang mengikuti gerakan.

## 🎯 Fitur Utama

- **Avatar Rendering**: Body, head, arms, legs dengan smooth animation
- **Body Orientation Detection**: Deteksi apakah body menghadap kiri/kanan/center
- **Head Orientation Detection**: Deteksi apakah head menengok kiri/kanan/center
- **Hand Tracking**: Render tangan berdasarkan MediaPipe hand landmarks
- **Eye Animation**: Avatar blink secara natural dengan random interval
- **Pose Smoothing**: Smooth landmark tracking untuk menghilangkan jitter

## 📁 Project Structure

```
PCV FP/
├── main.py                    # Main application loop - avatar rendering
├── config.py                  # Configuration - all settings & constants
├── asset_loader.py            # Load PNG images with alpha channel
├── image_processing.py        # Image overlay & blending operations
├── media_pipe_handler.py      # MediaPipe pose & hand detection wrappers
├── utils.py                   # Helper functions
└── img/                       # Asset folder
    ├── body_front.png
    ├── body_left.png, body_right.png
    ├── head.png
    ├── head_left.png, head_right.png
    ├── arm_upper.png, arm_lower.png
    ├── upper_leg_left/right.png
    ├── lower_leg_left/right.png
    └── background.png
```

## 🏗️ Architecture

### **main.py** - Core Application
- Initialize MediaPipe (Pose + Hands)
- Load camera input
- Main loop: detect → render → display
- Render pipeline:
  1. Pose landmarks detection
  2. Body orientation detection → select body image
  3. Head orientation detection → select head image
  4. Render body, head, arms, legs
  5. Hand detection & rendering
  6. Eye blinking animation
  7. Display dual windows (avatar + landmark debug)

### **config.py** - Configuration Hub
All settings in one place:
```python
# Display
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# MediaPipe thresholds
POSE_MIN_DETECTION_CONFIDENCE = 0.5
HANDS_MIN_DETECTION_CONFIDENCE = 0.5

# Body scaling
BODY_SCALE_MIN = 0.5
BODY_SCALE_MAX = 2.0

# Orientation detection (60% threshold)
# Body: 60% shoulder width needed to trigger left/right
# Head: 25% eye width needed to trigger left/right

# Animation
BLINK_CHANCE_PER_FRAME = 0.02  # 2% per frame
BLINK_DURATION_MIN = 1
BLINK_DURATION_MAX = 3
```

### **asset_loader.py** - Asset Management
- Load PNG images with transparency
- Auto-resize if needed
- Mirror left arm from right arm image
- Return dict of loaded images

### **image_processing.py** - Rendering Engine
Core functions:
- `overlay_rotated_limb(canvas, image, point1, point2, ...)`:
  - Calculate angle between two points
  - Rotate image to match limb angle
  - Scale based on distance
  - Alpha blend onto canvas
  
- `draw_head(canvas, head_img, anchor_pos, ...)`:
  - Position head relative to shoulder
  - Alpha blend with proper clipping
  
- `detect_body_orientation(left_shoulder, right_shoulder)`:
  - Compare shoulder X positions
  - Use percentage-based threshold (60% of shoulder width)
  - Apply hysteresis (5-frame buffer) for stability
  - Return 'left', 'right', or 'center'
  
- `detect_head_orientation(left_eye, right_eye, nose)`:
  - Compare nose X to eye center
  - Use percentage-based threshold (25% of eye width)
  - Apply hysteresis for stability
  - Return 'left', 'right', or 'center'

### **media_pipe_handler.py** - Detection Wrappers
- `PoseHandler`: Wrapper untuk MediaPipe Pose
- `HandsHandler`: Wrapper untuk MediaPipe Hands
- Simple interface untuk process frame

### **utils.py** - Helper Functions
- `calculate_body_scale()`: Calculate body size based on shoulder distance
- `get_point()`: Extract landmark position dengan validation

## 🎮 How It Works

1. **Camera Input**: 
   - Capture frame (1280x720)
   - Resize to 800x600
   - Flip horizontally (mirror effect)

2. **Pose Detection**:
   - MediaPipe detects 33 body landmarks
   - Get shoulder, hip, elbow, wrist positions
   - Extract eye positions for blinking

3. **Orientation Detection**:
   - Body: Compare left & right shoulder positions
   - Head: Compare nose position to eye center
   - Use hysteresis buffer (5 frames) untuk smooth detection

4. **Image Selection**:
   - Center: body_front, head.png
   - Left detection → show right image (inverted for camera mirror)
   - Right detection → show left image
   - Side views already have eyes pre-drawn

5. **Rendering**:
   - Overlay body image with rotation based on hip-to-shoulder angle
   - Overlay head at shoulder position
   - Overlay arms (upper + lower) with elbow-wrist angle
   - Overlay legs (upper + lower) with knee-ankle angle
   - Draw eyes only on center head view
   - All blending uses alpha channel

6. **Animation**:
   - Random blinking: 2% chance per frame
   - Blink duration: 1-3 frames

## 📊 Detection Parameters

### Body Orientation
- **Threshold**: 60% of shoulder width
- **Hysteresis**: 5-frame buffer with majority voting
- **Result**: 'left', 'right', or 'center'

### Head Orientation  
- **Threshold**: 25% of eye width
- **Hysteresis**: 5-frame buffer
- **Result**: 'left', 'right', or 'center'

### Body Scale
- Reference distance: 200 pixels
- Min scale: 0.5x
- Max scale: 2.0x
- Automatically adjusts based on shoulder distance

## 🖼️ Asset Images

All images are PNG with alpha channel (transparency):
- `body_front.png` (800x600): Front-facing body
- `body_left.png`, `body_right.png`: Side views
- `head.png`: Front face with eyes
- `head_left.png`, `head_right.png`: Side faces with eyes (pre-drawn)
- `arm_upper.png`, `arm_lower.png`: Arm segments
- `upper_leg_*.png`, `lower_leg_*.png`: Leg segments
- `background.png`: Optional background image

## 🔧 Configuration Tips

To adjust sensitivity:
- **More responsive**: Lower threshold percentages in config.py
- **More stable**: Increase BUFFER_SIZE in image_processing.py
- **Bigger avatar**: Increase BODY_SCALE_MAX
- **Faster blinking**: Increase BLINK_CHANCE_PER_FRAME

## 🎬 Running the Application

```bash
python main.py
```

Press `q` to quit.

Two windows will appear:
1. **Avatar Window**: VTuber character rendering
2. **Landmark Window**: Debug view with pose/hand landmarks

## 📦 Dependencies

- opencv-python (cv2)
- mediapipe
- numpy

## 🎨 Design Principles

1. **Modular**: Each file has clear responsibility
2. **Configurable**: All settings in config.py
3. **Stable**: Hysteresis & smoothing for jitter-free tracking
4. **Efficient**: Vectorized numpy operations
5. **Flexible**: Easy to swap images or add new features
- **Clean interface untuk akses MediaPipe**

### **utils.py** - Helper Functions
- `smooth_landmark()` - EMA smoothing untuk stabilitas
- `calculate_body_scale()` - Hitung scale dari shoulder distance
- `get_point()` - Get landmark dengan smoothing
- **Fungsi-fungsi utility yang bisa di-reuse**

### **main.py** - Core Loop (267 lines saja!)
- Import semua modules
- Load assets & initialize
- Main loop yang bersih dan organized
- Setiap section punya komentar jelas
- **Lebih mudah di-debug karena ringkas**

## 🔧 Cara Modify

### Mau ubah setting?
Edit `config.py`:
```python
# Contoh: ubah warna brush
BRUSH_BRISTLE_COLOR = (100, 100, 255)  # Ubah RGB di sini
```

### Mau ubah logika rendering body?
Edit `main.py` bagian `# ===== HEAD RENDERING =====` atau `# ===== BODY RENDERING =====`

### Mau optimize image processing?
Edit `image_processing.py` function `overlay_rotated_limb()`

### Mau ubah smoothing?
Edit `config.py`:
```python
LANDMARK_SMOOTHING_FACTOR = 0.7  # Lower = lebih responsive
```

## 🚀 Run Program
```bash
python main.py
```

## 📊 Performance Tips

1. **Reduce Confidence** di `config.py`:
   - `POSE_MIN_DETECTION_CONFIDENCE = 0.15` (lebih cepat, tapi kurang akurat)
   
2. **Reduce Resolution**:
   - `WINDOW_WIDTH = 640`, `WINDOW_HEIGHT = 480` (lebih cepat)
   
3. **Disable Drawing Canvas** (jika gak perlu):
   - Comment bagian `# ===== BRUSH RENDERING =====`
   
4. **Skip Frames** (jika performance butuh):
   - Tambah counter untuk skip frame processing

## 📝 Notes

- Total kode **661 lines** (vs 583 lines lama, tapi lebih organized!)
- Setiap file **under 300 lines** - mudah di-debug
- **Clear separation of concerns** - tiap file punya job spesifik
- **Easy to test** - fungsi-fungsi bisa di-test separate
- **Easy to modify** - tau mau ubah apa, edit file mana
