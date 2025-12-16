"""
VTuber Avatar - Main Program with Dual Windows
- Window 1: Landmark visualization (pose + hand landmarks)
- Window 2: Avatar rendering (full character)
"""
import cv2
import numpy as np
import random

# Import dari modules
import config
from asset_loader import load_assets, generate_eye_image
from media_pipe_handler import PoseHandler, HandsHandler
from image_processing import overlay_rotated_limb, draw_head, detect_body_orientation, detect_head_orientation
from utils import calculate_body_scale, get_point

# ========== LOAD ASSETS ========== 
images = load_assets()
images['eye'] = generate_eye_image()

# Load background image if available
import os
background_img = None
bg_path = os.path.join('img', 'background.png')
if os.path.exists(bg_path):
    bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg_img is not None:
        background_img = cv2.resize(bg_img, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
        print('Background image loaded.')
    else:
        print('Failed to load background image.')
else:
    print('No background image found, using solid color.')

# ========== INITIALIZE MEDIAPIPE ==========
pose_handler = PoseHandler()
hands_handler = HandsHandler()

# ========== INITIALIZE CAMERA ==========
cap = cv2.VideoCapture(config.CAMERA_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)

# ========== ANIMATION STATE ==========
is_blinking = False
blink_timer = 0
blink_duration = 0

# ========== SMOOTHING BUFFERS ==========
landmark_buffer = {}

# ========== CANVAS STATE ==========
canvas_template = None
drawing_canvas = None
# ========== MAIN LOOP ==========
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # ===== FRAME PREPARATION =====
    frame = cv2.resize(frame, (config.WINDOW_WIDTH, config.WINDOW_HEIGHT), 
                       interpolation=cv2.INTER_LINEAR)
    frame = cv2.flip(frame, 1)
    h, w = config.WINDOW_HEIGHT, config.WINDOW_WIDTH
    
    # ===== CANVAS INITIALIZATION =====
    if canvas_template is None:
        if background_img is not None:
            canvas_template = background_img.copy()
        else:
            canvas_template = np.zeros((h, w, 3), dtype=np.uint8)
            canvas_template[:] = config.BG_COLOR
        drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)  # Semua hitam/kosong
    
    canvas = canvas_template.copy()
    
    # ===== BLEND DRAWING =====
    # Cek apakah ada garis yang sudah digambar (pixel yang bukan hitam)
    mask = np.any(drawing_canvas != 0, axis=2)
    if np.any(mask):
        cv2.addWeighted(canvas, 1.0 - config.DRAWING_BLEND_RATIO, 
                       drawing_canvas, config.DRAWING_BLEND_RATIO, 0, canvas)
    
    # ===== PROCESS IMAGES =====
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose_handler.process(rgb_frame)
    hand_results = hands_handler.process(rgb_frame)
    
    # ===== DRAW LANDMARKS ON FRAME (untuk window 1) =====
    landmark_frame = frame.copy()
    
    # Draw pose landmarks
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            x = int(landmark.x * config.WINDOW_WIDTH)
            y = int(landmark.y * config.WINDOW_HEIGHT)
            if 0 <= x < landmark_frame.shape[1] and 0 <= y < landmark_frame.shape[0]:
                cv2.circle(landmark_frame, (x, y), 3, (0, 255, 0), -1)  # Green dots untuk pose
    
    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * config.WINDOW_WIDTH)
                y = int(landmark.y * config.WINDOW_HEIGHT)
                if 0 <= x < landmark_frame.shape[1] and 0 <= y < landmark_frame.shape[0]:
                    cv2.circle(landmark_frame, (x, y), 2, (255, 0, 0), -1)  # Blue dots untuk tangan
    
    # ===== UPDATE BLINK ANIMATION - RANDOM BLINKING =====
    if random.random() < config.BLINK_CHANCE_PER_FRAME:
        is_blinking = True
        blink_duration = random.randint(config.BLINK_DURATION_MIN, config.BLINK_DURATION_MAX)
    
    if is_blinking:
        blink_timer += 1
        if blink_timer >= blink_duration:
            is_blinking = False
            blink_timer = 0
    
    # ===== RESET HAND POSITIONS =====
    hand_positions = {}
    
    # ========== POSE RENDERING ==========
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        # Helper function untuk get point dengan smoothing
        def get_pt(idx):
            return get_point(idx, landmarks, w, h, landmark_buffer, config.LANDMARK_SMOOTHING_FACTOR)
        
        # Hitung body scale
        left_shoulder = get_pt(11)
        right_shoulder = get_pt(12)
        body_scale = calculate_body_scale(
            left_shoulder, right_shoulder,
            config.SHOULDER_DISTANCE_REFERENCE,
            config.BODY_SCALE_MIN,
            config.BODY_SCALE_MAX
        )
        
        # ===== HEAD RENDERING =====
        left_eye = get_pt(2)
        right_eye = get_pt(5)
        nose = get_pt(0)
        
        if nose and left_eye and right_eye and left_shoulder and right_shoulder:
            # Deteksi orientasi kepala
            head_orientation = detect_head_orientation(left_eye, right_eye, nose)
            
            # Pilih gambar kepala berdasarkan orientasi (SWAP: left detection = right image)
            if head_orientation == 'left':
                head_img = images.get('head_right', images.get('head'))
            elif head_orientation == 'right':
                head_img = images.get('head_left', images.get('head'))
            else:
                head_img = images.get('head')
            
            # Debug: cek if head_img valid
            if head_img is None:
                print(f"WARNING: head_img is None! Orientation: {head_orientation}")
            
            # Gunakan mid_shoulder sebagai anchor untuk head positioning (lebih stabil)
            mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                           (left_shoulder[1] + right_shoulder[1]) // 2)
            
            canvas = draw_head(canvas, head_img, mid_shoulder, left_eye, right_eye,
                             body_scale, config.HEAD_SCALE_MULTIPLIER, 
                             config.HEAD_POSITION_Y_OFFSET)
            
            # Draw mata HANYA jika tidak blinking DAN head menghadap CENTER
            # Untuk side-view, mata sudah ada di image, jangan gambar lagi
            if not is_blinking and head_orientation == 'center':
                eye_scale = int(body_scale * config.EYE_SCALE_MULTIPLIER)
                cv2.circle(canvas, left_eye, eye_scale, (0, 0, 0, 255), -1)
                cv2.circle(canvas, right_eye, eye_scale, (0, 0, 0, 255), -1)
        
        # ===== BODY RENDERING =====
        left_hip = get_pt(23)
        right_hip = get_pt(24)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            # Deteksi orientasi body
            body_orientation = detect_body_orientation(left_shoulder, right_shoulder)
            
            # Pilih gambar body berdasarkan orientasi (SWAP: left detection = right image)
            if body_orientation == 'left':
                body_img = images.get('body_right', images.get('body'))
            elif body_orientation == 'right':
                body_img = images.get('body_left', images.get('body'))
            else:
                body_img = images.get('body')
            
            # Debug: cek if body_img valid
            if body_img is None:
                print(f"WARNING: body_img is None! Orientation: {body_orientation}")
            
            mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                           (left_shoulder[1] + right_shoulder[1]) // 2)
            mid_hip = ((left_hip[0] + right_hip[0]) // 2,
                      (left_hip[1] + right_hip[1]) // 2)
            
            canvas = overlay_rotated_limb(canvas, body_img, mid_shoulder, mid_hip,
                                         scale_width=config.BODY_SCALE_WIDTH)
        
        # ===== RIGHT ARM RENDERING =====
        right_elbow = get_pt(14)
        right_wrist = get_pt(16)
        if right_shoulder and right_elbow:
            canvas = overlay_rotated_limb(canvas, images.get('right_arm_upper'),
                                         right_shoulder, right_elbow,
                                         scale_width=config.ARM_SCALE_WIDTH)
        if right_elbow and right_wrist:
            canvas = overlay_rotated_limb(canvas, images.get('right_arm_lower'),
                                         right_elbow, right_wrist,
                                         scale_width=config.ARM_SCALE_WIDTH)
        
        # ===== LEFT ARM RENDERING =====
        left_elbow = get_pt(13)
        left_wrist = get_pt(15)
        if left_shoulder and left_elbow:
            canvas = overlay_rotated_limb(canvas, images.get('left_arm_upper'),
                                         left_shoulder, left_elbow,
                                         scale_width=config.ARM_SCALE_WIDTH)
        if left_elbow and left_wrist:
            canvas = overlay_rotated_limb(canvas, images.get('left_arm_lower'),
                                         left_elbow, left_wrist,
                                         scale_width=config.ARM_SCALE_WIDTH)
        
        # ===== RIGHT LEG RENDERING =====
        right_knee = get_pt(26)
        right_ankle = get_pt(28)
        if right_hip and right_knee:
            canvas = overlay_rotated_limb(canvas, images.get('right_leg_upper'),
                                         right_hip, right_knee)
        if right_knee and right_ankle:
            canvas = overlay_rotated_limb(canvas, images.get('right_leg_lower'),
                                         right_knee, right_ankle)
        
        # ===== LEFT LEG RENDERING =====
        left_knee = get_pt(25)
        left_ankle = get_pt(27)
        if left_hip and left_knee:
            canvas = overlay_rotated_limb(canvas, images.get('left_leg_upper'),
                                         left_hip, left_knee)
        if left_knee and left_ankle:
            canvas = overlay_rotated_limb(canvas, images.get('left_leg_lower'),
                                         left_knee, left_ankle)
    
    # ========== HAND RENDERING ==========
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                             hand_results.multi_handedness):
            # ===== PALM DRAWING =====
            palm_points = [0, 5, 9, 13, 17]
            palm_coords = []
            
            for idx in palm_points:
                landmark = hand_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    palm_coords.append([x, y])
            
            if len(palm_coords) >= 3:
                palm_coords_array = np.array(palm_coords, dtype=np.int32)
                thickness = max(2, int(body_scale * config.HAND_PALM_THICKNESS_MULTIPLIER))
                cv2.polylines(canvas, [palm_coords_array], True, (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.fillPoly(canvas, [palm_coords_array], (0, 0, 0))
            
            # ===== WRIST DRAWING =====
            wrist = hand_landmarks.landmark[0]
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))
            wrist_radius = max(8, int(body_scale * config.HAND_WRIST_RADIUS_MULTIPLIER))
            cv2.circle(canvas, wrist_pos, wrist_radius, (0, 0, 0), -1)
            cv2.circle(canvas, wrist_pos, wrist_radius, (0, 0, 0),
                      max(1, int(body_scale * 2)), cv2.LINE_AA)
            
            # ===== FINGER DRAWING =====
            finger_joints = [(2, 4), (6, 8), (10, 12), (14, 16), (18, 20)]
            finger_color = (0, 0, 0)
            finger_thickness = max(2, int(body_scale * config.FINGER_THICKNESS_MULTIPLIER))
            palm_center = wrist_pos
            
            for mcp_idx, tip_idx in finger_joints:
                mcp = hand_landmarks.landmark[mcp_idx]
                pip = hand_landmarks.landmark[mcp_idx + 1]
                tip = hand_landmarks.landmark[tip_idx]
                
                mcp_pos = (int(mcp.x * w), int(mcp.y * h))
                pip_pos = (int(pip.x * w), int(pip.y * h))
                tip_pos = (int(tip.x * w), int(tip.y * h))
                
                # Draw PALM -> MCP
                if (0 <= palm_center[0] < canvas.shape[1] and 0 <= palm_center[1] < canvas.shape[0] and
                    0 <= mcp_pos[0] < canvas.shape[1] and 0 <= mcp_pos[1] < canvas.shape[0]):
                    cv2.line(canvas, palm_center, mcp_pos, finger_color, finger_thickness, cv2.LINE_AA)
                
                # Draw MCP -> PIP
                if (0 <= mcp_pos[0] < canvas.shape[1] and 0 <= mcp_pos[1] < canvas.shape[0] and
                    0 <= pip_pos[0] < canvas.shape[1] and 0 <= pip_pos[1] < canvas.shape[0]):
                    cv2.line(canvas, mcp_pos, pip_pos, finger_color, finger_thickness, cv2.LINE_AA)
                
                # Draw PIP -> TIP
                if (0 <= pip_pos[0] < canvas.shape[1] and 0 <= pip_pos[1] < canvas.shape[0] and
                    0 <= tip_pos[0] < canvas.shape[1] and 0 <= tip_pos[1] < canvas.shape[0]):
                    cv2.line(canvas, pip_pos, tip_pos, finger_color, finger_thickness, cv2.LINE_AA)
                    cv2.circle(canvas, tip_pos, int(finger_thickness * config.FINGER_TIP_RADIUS_MULTIPLIER),
                              finger_color, -1)
            
            hand_label = handedness.classification[0].label
            hand_positions[hand_label] = palm_center
    
    # ========== DISPLAY ==========
    cv2.imshow('Avatar - VTuber Character', canvas)
    cv2.imshow('Landmark Scan - Pose & Hands', landmark_frame)
    
    # ========== FRAME RATE CONTROL ==========
    key = cv2.waitKey(config.FRAME_TIME_MS) & 0xFF
    if key == ord('q'):
        break

# ========== CLEANUP ==========
cap.release()
cv2.destroyAllWindows()
pose_handler.close()
hands_handler.close()

print("Program selesai!")
