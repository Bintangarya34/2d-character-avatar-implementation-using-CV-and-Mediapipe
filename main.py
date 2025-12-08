import cv2
import mediapipe as mp
import numpy as np
import math
import os
import random

# --- 1. CONFIG & LOAD ASSETS ---
# Pastikan nama file sesuai dengan screenshotmu
assets = {
    'body': 'body.png',
    'head': 'head.png',
    'right_arm_upper': 'arm_upper.png', # Asumsi ini tangan kanan
    'right_arm_lower': 'arm_lower.png',
    'left_arm_upper': 'arm_upper.png',  # Kita pakai gambar sama, nanti di-flip kalau perlu
    'left_arm_lower': 'arm_lower.png',
    'right_leg_upper': 'upper_leg_right.png',
    'right_leg_lower': 'lower_leg_right.png',
    'left_leg_upper': 'upper_leg_left.png',
    'left_leg_lower': 'lower_leg_left.png',
}

images = {}

print("Loading Assets...")
for key, filename in assets.items():
    filepath = os.path.join('img', filename)
    if os.path.exists(filepath):
        # IMREAD_UNCHANGED penting agar transparansi (Alpha channel) terbaca
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: Gagal load {filepath}")
        else:
            images[key] = img
            print(f"Loaded {filepath}")
            
            # Mirror gambar lengan kiri dari kanan
            if 'left_arm' in key:
                images[key] = cv2.flip(img, 1)  # Flip horizontal
    else:
        print(f"Warning: File {filepath} tidak ditemukan!")

# --- 2. FUNGSI MATEMATIKA CITRA (CORE IMAGE PROCESSING) ---

def overlay_rotated_limb(background, img_part, point_a, point_b, scale_width=1.0, angle_offset=90):
    """
    Menempelkan gambar anggota tubuh di antara Titik A dan Titik B.
    Gambar diasumsikan vertikal (berdiri tegak) pada aslinya.
    angle_offset: offset sudut rotasi (default 90 untuk lengan/kaki, 0 untuk kepala)
    """
    if img_part is None: return background
    
    # 1. Hitung Jarak (Euclidean Distance) untuk Scaling Height
    # Agar panjang gambar lengan mengikuti panjang lengan asli kita di kamera
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    limb_length = np.sqrt(dx**2 + dy**2)
    
    if limb_length == 0: return background

    # Ambil dimensi asli gambar
    h_img, w_img = img_part.shape[:2]
    
    # 2. Scaling
    # Kita ubah tinggi gambar agar sama dengan jarak sendi (limb_length)
    # Lebar menyesuaikan rasio agar tidak gepeng
    ratio = w_img / h_img
    new_h = int(limb_length)
    new_w = int(new_h * ratio * scale_width) 
    
    if new_w <= 0 or new_h <= 0: return background
    
    resized_img = cv2.resize(img_part, (new_w, new_h))
    
    # 3. Hitung Sudut Rotasi
    # atan2 menghasilkan sudut dalam radian antara garis dan sumbu X
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) - angle_offset # offset untuk rotasi

    # 4. Rotasi Gambar (Affine Transform)
    # Kita perlu kanvas yang lebih besar untuk menampung hasil putaran biar gak terpotong
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0) # -angle karena koordinat Y OpenCV kebalik
    
    # Hitung bounding box baru setelah rotasi
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w_rot = int((new_h * sin) + (new_w * cos))
    new_h_rot = int((new_h * cos) + (new_w * sin))
    
    M[0, 2] += (new_w_rot / 2) - center[0]
    M[1, 2] += (new_h_rot / 2) - center[1]
    
    rotated_img = cv2.warpAffine(resized_img, M, (new_w_rot, new_h_rot), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # 5. Penempatan (Positioning)
    # Kita taruh titik tengah gambar hasil rotasi di titik tengah antara A dan B
    mid_x = int((point_a[0] + point_b[0]) / 2)
    mid_y = int((point_a[1] + point_b[1]) / 2)
    
    y1, y2 = mid_y - new_h_rot // 2, mid_y + new_h_rot // 2
    x1, x2 = mid_x - new_w_rot // 2, mid_x + new_w_rot // 2
    
    # 6. Alpha Blending (Tempel Gambar)
    # Cek batas frame agar tidak error
    bg_h, bg_w = background.shape[:2]
    
    # Clipping koordinat
    y1_clip = max(0, y1)
    y2_clip = min(bg_h, y2)
    x1_clip = max(0, x1)
    x2_clip = min(bg_w, x2)
    
    # Hitung offset di overlay image
    dy1 = max(0, -y1)
    dx1 = max(0, -x1)
    dy2 = dy1 + (y2_clip - y1_clip)
    dx2 = dx1 + (x2_clip - x1_clip)
    
    if y1_clip >= y2_clip or x1_clip >= x2_clip: 
        return background

    # Ambil region of interest dengan ukuran yang sama
    overlay_crop = rotated_img[dy1:dy2, dx1:dx2]
    background_crop = background[y1_clip:y2_clip, x1_clip:x2_clip]
    
    # Pastikan ukuran sama
    if overlay_crop.shape[0] != background_crop.shape[0] or overlay_crop.shape[1] != background_crop.shape[1]:
        overlay_crop = cv2.resize(overlay_crop, (background_crop.shape[1], background_crop.shape[0]))
    
    # Rumus Blending
    if overlay_crop.shape[2] == 4:  # Ada alpha channel
        alpha_s = overlay_crop[:, :, 3] / 255.0
        alpha_s = np.expand_dims(alpha_s, axis=2)  # Expand untuk broadcasting
        alpha_l = 1.0 - alpha_s
        
        for c in range(0, 3):
            background[y1_clip:y2_clip, x1_clip:x2_clip, c] = (alpha_s[:, :, 0] * overlay_crop[:, :, c] +
                                                               alpha_l[:, :, 0] * background_crop[:, :, c]).astype(np.uint8)
    else:  # Tidak ada alpha channel
        for c in range(0, 3):
            background[y1_clip:y2_clip, x1_clip:x2_clip, c] = overlay_crop[:, :, c]
                                                           
    return background

# --- 3. MAIN LOOP MEDIAPIPE ---

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

# Background color
bg_color = [255, 255, 255]  # Putih

# Animasi blink
blink_timer = 0
is_blinking = False
blink_duration = 0

# Kuas state (panjang, perlu kedua tangan)
brush_handle_length = 300  # Panjang gagang kuas
brush_head_length = 80     # Panjang bulu kuas
brush_width = 20           # Lebar gagang
brush_color = (200, 150, 100)  # Warna gagang (cokelat muda - BGR)
brush_bristle_color = (100, 100, 255)  # Warna bulu (merah muda)
brush_grabbed = False
brush_prev_tip = None  # Untuk tracking garis (smoothing posisi)

# Hand positions untuk menggengam kuas (kedua tangan)
brush_hand1_pos = None  # Tangan 1 (base gagang)
brush_hand2_pos = None  # Tangan 2 (ujung gagang)

# Smoothing factor (0.0 = no smoothing, 1.0 = complete smoothing)
# Lower value = more responsive, higher value = more smooth
smoothing_factor = 0.7

# Landmark smoothing buffers
landmark_buffer = {}
hand_landmark_buffer = {}

# Drawing canvas (untuk track coretan pensil)
drawing_canvas = None

# Generate simple eye image (black circle)
eye_size = 30
eye_img = np.zeros((eye_size, eye_size, 4), dtype=np.uint8)
cv2.circle(eye_img, (eye_size//2, eye_size//2), eye_size//2 - 2, (0, 0, 0, 255), -1)
images['eye'] = eye_img

def smooth_value(current, previous, factor):
    """Apply exponential moving average smoothing"""
    if previous is None:
        return current
    return previous * factor + current * (1 - factor)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Resize frame ke resolusi yang lebih besar tapi masih smooth
    frame = cv2.resize(frame, (960, 720))
    
    # Mirror frame input untuk efek cermin
    frame = cv2.flip(frame, 1)
    
    # Canvas dengan background color
    h, w, _ = frame.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8)
    canvas[:] = bg_color
    
    # Initialize drawing canvas jika belum ada
    if drawing_canvas is None:
        drawing_canvas = np.ones((h, w, 3), dtype=np.uint8)
        drawing_canvas[:] = bg_color
    
    # Overlay drawing canvas ke canvas utama
    canvas = cv2.addWeighted(canvas, 0.7, drawing_canvas, 0.3, 0)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    # Process hands
    hand_results = hands.process(rgb_frame)
    
    # Update blink animation
    if random.random() < 0.02:  # 2% chance blink setiap frame
        is_blinking = True
        blink_duration = random.randint(1, 3)  # Blink selama 1-3 frame
    
    if is_blinking:
        blink_timer += 1
        if blink_timer >= blink_duration:
            is_blinking = False
            blink_timer = 0
    
    # Reset hand positions untuk frame baru
    brush_hand1_pos = None
    brush_hand2_pos = None
    hand_positions = {}  # Dict untuk track posisi tangan berdasarkan label
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Convert canvas to BGRA untuk alpha blending
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
        
        # Landmark indices (dari MediaPipe Pose)
        # 11=left_shoulder, 12=right_shoulder, 13=left_elbow, 14=right_elbow
        # 15=left_wrist, 16=right_wrist, 23=left_hip, 24=right_hip
        # 25=left_knee, 26=right_knee, 27=left_ankle, 28=right_ankle
        # 0=nose, 1=left_eye_inner, dll
        
        def get_point(idx):
            if idx < len(landmarks) and landmarks[idx].visibility > 0.5:
                return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))
            return None
        
        # Hitung body_scale berdasarkan shoulder distance (untuk consistency)
        left_shoulder = get_point(11)
        right_shoulder = get_point(12)
        
        body_scale = 1.0
        if left_shoulder and right_shoulder:
            shoulder_distance = np.sqrt((right_shoulder[0] - left_shoulder[0])**2 + (right_shoulder[1] - left_shoulder[1])**2)
            body_scale = shoulder_distance / 200.0  # Normalize ke ratio tertentu
        
        # Overlay Body Parts
        # Head (letakkan di atas nose, skala sesuai body)
        left_eye = get_point(2)  # left_eye
        right_eye = get_point(5)  # right_eye
        nose = get_point(0)
        if nose and left_eye and right_eye:
            head_img = images.get('head')
            if head_img is not None:
                h_img, w_img = head_img.shape[:2]
                # Scale head berdasarkan body_scale
                new_w = int(w_img * body_scale * 1.5)  # 1.5x multiplier untuk ukuran kepala
                new_h = int(h_img * body_scale * 1.5)
                
                if new_w > 0 and new_h > 0:
                    resized_head = cv2.resize(head_img, (new_w, new_h))
                    
                    # Posisi kepala di atas nose
                    y1 = int(nose[1] - new_h * 0.6)
                    x1 = int(nose[0] - new_w // 2)
                    y2 = y1 + new_h
                    x2 = x1 + new_w
                    
                    # Clipping
                    y1_clip = max(0, y1)
                    y2_clip = min(canvas.shape[0], y2)
                    x1_clip = max(0, x1)
                    x2_clip = min(canvas.shape[1], x2)
                    
                    if y1_clip < y2_clip and x1_clip < x2_clip:
                        dy1 = max(0, -y1)
                        dx1 = max(0, -x1)
                        dy2 = dy1 + (y2_clip - y1_clip)
                        dx2 = dx1 + (x2_clip - x1_clip)
                        
                        overlay_crop = resized_head[dy1:dy2, dx1:dx2]
                        if overlay_crop.shape[0] == (y2_clip - y1_clip) and overlay_crop.shape[1] == (x2_clip - x1_clip):
                            if resized_head.shape[2] == 4:
                                alpha_s = overlay_crop[:, :, 3] / 255.0
                                alpha_s = np.expand_dims(alpha_s, axis=2)
                                for c in range(0, 3):
                                    canvas[y1_clip:y2_clip, x1_clip:x2_clip, c] = (alpha_s[:, :, 0] * overlay_crop[:, :, c] + 
                                                                                   (1.0 - alpha_s[:, :, 0]) * canvas[y1_clip:y2_clip, x1_clip:x2_clip, c]).astype(np.uint8)
                    
                    # Draw mata (mata berkedip)
                    if not is_blinking:
                        eye_scale = int(body_scale * 15)  # Ukuran mata berdasarkan body scale
                        
                        # Gunakan actual eye landmarks untuk positioning yang lebih stabil
                        # Hitung posisi mata berdasarkan landmark sebenarnya, bukan persentase
                        if left_eye and right_eye:
                            # Mata akan ditampilkan di posisi landmark asli, tapi hanya kalau ada kepala
                            left_eye_pos = left_eye
                            right_eye_pos = right_eye
                            
                            # Draw mata sebagai lingkaran hitam
                            cv2.circle(canvas, left_eye_pos, eye_scale, (0, 0, 0, 255), -1)
                            cv2.circle(canvas, right_eye_pos, eye_scale, (0, 0, 0, 255), -1)
        
        # Body (dari mid-shoulders ke mid-hips)
        left_hip = get_point(23)
        right_hip = get_point(24)
        
        if left_shoulder and right_shoulder and left_hip and right_hip:
            mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2, 
                           (left_shoulder[1] + right_shoulder[1]) // 2)
            mid_hip = ((left_hip[0] + right_hip[0]) // 2, 
                       (left_hip[1] + right_hip[1]) // 2)
            canvas = overlay_rotated_limb(canvas, images.get('body'), mid_shoulder, mid_hip, scale_width=1.2)
        
        # Right Arm
        right_elbow = get_point(14)
        right_wrist = get_point(16)
        if right_shoulder and right_elbow:
            canvas = overlay_rotated_limb(canvas, images.get('right_arm_upper'), right_shoulder, right_elbow, scale_width=1.5)
        if right_elbow and right_wrist:
            canvas = overlay_rotated_limb(canvas, images.get('right_arm_lower'), right_elbow, right_wrist, scale_width=1.5)
        
        # Left Arm
        left_elbow = get_point(13)
        left_wrist = get_point(15)
        if left_shoulder and left_elbow:
            canvas = overlay_rotated_limb(canvas, images.get('left_arm_upper'), left_shoulder, left_elbow, scale_width=1.5)
        if left_elbow and left_wrist:
            canvas = overlay_rotated_limb(canvas, images.get('left_arm_lower'), left_elbow, left_wrist, scale_width=1.5)
        
        # Right Leg
        right_hip = get_point(24)
        right_knee = get_point(26)
        right_ankle = get_point(28)
        if right_hip and right_knee:
            canvas = overlay_rotated_limb(canvas, images.get('right_leg_upper'), right_hip, right_knee)
        if right_knee and right_ankle:
            canvas = overlay_rotated_limb(canvas, images.get('right_leg_lower'), right_knee, right_ankle)
        
        # Left Leg
        left_hip = get_point(23)
        left_knee = get_point(25)
        left_ankle = get_point(27)
        if left_hip and left_knee:
            canvas = overlay_rotated_limb(canvas, images.get('left_leg_upper'), left_hip, left_knee)
        if left_knee and left_ankle:
            canvas = overlay_rotated_limb(canvas, images.get('left_leg_lower'), left_knee, left_ankle)
        
        # Convert kembali ke BGR
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)
    
    # Draw hands dengan jari yang panjang
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            # Palm landmarks: 0=wrist, 5=index_mcp, 9=middle_mcp, 13=ring_mcp, 17=pinky_mcp
            palm_points = [0, 5, 9, 13, 17]
            palm_coords = []
            
            for idx in palm_points:
                landmark = hand_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
                    palm_coords.append([x, y])
            
            # Draw palm sebagai polygon
            if len(palm_coords) >= 3:
                palm_coords_array = np.array(palm_coords, dtype=np.int32)
                cv2.polylines(canvas, [palm_coords_array], True, (0, 0, 0), max(2, int(body_scale * 3)), cv2.LINE_AA)
                cv2.fillPoly(canvas, [palm_coords_array], (200, 200, 200))  # Fill dengan warna abu-abu muda
            
            # Get wrist position dan buat rounded
            wrist = hand_landmarks.landmark[0]
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))
            wrist_radius = max(8, int(body_scale * 12))  # Rounded wrist
            cv2.circle(canvas, wrist_pos, wrist_radius, (200, 200, 200), -1)  # Wrist bulat
            cv2.circle(canvas, wrist_pos, wrist_radius, (0, 0, 0), max(1, int(body_scale * 2)), cv2.LINE_AA)  # Border
            
            # Finger tips dan joints untuk draw lengkap
            # Thumb: 2=MCP, 4=tip
            # Index: 6=PIP, 8=tip
            # Middle: 10=PIP, 12=tip
            # Ring: 14=PIP, 16=tip
            # Pinky: 18=PIP, 20=tip
            
            finger_joints = [
                (2, 4),    # Thumb
                (6, 8),    # Index
                (10, 12),  # Middle
                (14, 16),  # Ring
                (18, 20)   # Pinky
            ]
            
            finger_color = (0, 0, 0)  # Hitam
            finger_thickness = max(2, int(body_scale * 4))
            
            # Hitung distance untuk grabbing - KUAS PERLU KEDUA TANGAN
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
            
            # Hitung rata-rata posisi tangan (palm center)
            palm_center = ((thumb_pos[0] + index_pos[0] + middle_pos[0]) // 3,
                          (thumb_pos[1] + index_pos[1] + middle_pos[1]) // 3)
            
            # Track kedua tangan untuk kuas
            hand_label = handedness.classification[0].label  # "Right" atau "Left"
            hand_positions[hand_label] = palm_center
            
            # Assign ke brush hands based on label
            if hand_label == "Right":
                brush_hand1_pos = palm_center
            else:
                brush_hand2_pos = palm_center
            
            # Cek apakah kedua tangan sudah menggengam kuas
            if brush_hand1_pos and brush_hand2_pos:
                brush_grabbed = True
                # Jarak kedua tangan = arah & panjang kuas
                dx = brush_hand2_pos[0] - brush_hand1_pos[0]
                dy = brush_hand2_pos[1] - brush_hand1_pos[1]
                hand_distance = np.sqrt(dx**2 + dy**2)
                
                if hand_distance > 30:  # Minimal jarak untuk dianggap grab
                    # Normalize vektor arah
                    norm_dx = dx / hand_distance
                    norm_dy = dy / hand_distance
                    
                    # Ujung kuas di depan tangan 2
                    brush_tip = (int(brush_hand2_pos[0] + norm_dx * brush_head_length),
                               int(brush_hand2_pos[1] + norm_dy * brush_head_length))
                    
                    # Simpan posisi sebelumnya untuk garis (line tracking)
                    if brush_prev_tip is not None:
                        # Draw garis dari posisi sebelumnya ke posisi sekarang
                        if (0 <= brush_tip[0] < drawing_canvas.shape[1] and 0 <= brush_tip[1] < drawing_canvas.shape[0] and
                            0 <= brush_prev_tip[0] < drawing_canvas.shape[1] and 0 <= brush_prev_tip[1] < drawing_canvas.shape[0]):
                            cv2.line(drawing_canvas, brush_prev_tip, brush_tip, brush_bristle_color, 15, cv2.LINE_AA)
                    else:
                        # First touch - set prev_tip
                        brush_prev_tip = brush_tip
                    
                    brush_prev_tip = brush_tip
                else:
                    brush_grabbed = False
                    brush_prev_tip = None
            else:
                brush_grabbed = False
                brush_prev_tip = None
            
            for mcp_idx, tip_idx in finger_joints:
                mcp = hand_landmarks.landmark[mcp_idx]
                tip = hand_landmarks.landmark[tip_idx]
                
                mcp_pos = (int(mcp.x * w), int(mcp.y * h))
                tip_pos = (int(tip.x * w), int(tip.y * h))
                
                # Draw jari dari MCP ke TIP
                if (0 <= mcp_pos[0] < canvas.shape[1] and 0 <= mcp_pos[1] < canvas.shape[0] and
                    0 <= tip_pos[0] < canvas.shape[1] and 0 <= tip_pos[1] < canvas.shape[0]):
                    cv2.line(canvas, mcp_pos, tip_pos, finger_color, finger_thickness, cv2.LINE_AA)
                    # Lingkaran di tip
                    cv2.circle(canvas, tip_pos, int(finger_thickness * 1.2), finger_color, -1)
    
    # Draw KUAS (panjang, perlu kedua tangan) - render dari hand positions yang sudah di-track
    if len(hand_positions) >= 2:
        # Ambil dari dict
        right_hand = hand_positions.get("Right")
        left_hand = hand_positions.get("Left")
        
        if right_hand and left_hand:
            brush_hand1_pos = right_hand
            brush_hand2_pos = left_hand
            
            # Gagang kuas dari tangan 1 ke tangan 2
            cv2.line(canvas, brush_hand1_pos, brush_hand2_pos, brush_color, brush_width, cv2.LINE_AA)
            
            # Hitung arah kuas
            dx = brush_hand2_pos[0] - brush_hand1_pos[0]
            dy = brush_hand2_pos[1] - brush_hand1_pos[1]
            hand_distance = np.sqrt(dx**2 + dy**2)
            
            if hand_distance > 30:
                # Normalize & perpanjang ke ujung kuas (bulu)
                norm_dx = dx / hand_distance
                norm_dy = dy / hand_distance
                
                brush_tip = (int(brush_hand2_pos[0] + norm_dx * brush_head_length),
                            int(brush_hand2_pos[1] + norm_dy * brush_head_length))
                
                # Draw bulu kuas (dari tangan 2 ke ujung)
                cv2.line(canvas, brush_hand2_pos, brush_tip, brush_bristle_color, brush_width - 5, cv2.LINE_AA)
                
                # Ujung kuas (bulat)
                cv2.circle(canvas, brush_tip, brush_width // 2, brush_bristle_color, -1)
                
                # Gagang base (bulat)
                cv2.circle(canvas, brush_hand1_pos, brush_width // 2, brush_color, -1)
                
                # Update drawing saat kedua tangan grab
                if brush_prev_tip is not None:
                    if (0 <= brush_tip[0] < drawing_canvas.shape[1] and 0 <= brush_tip[1] < drawing_canvas.shape[0] and
                        0 <= brush_prev_tip[0] < drawing_canvas.shape[1] and 0 <= brush_prev_tip[1] < drawing_canvas.shape[0]):
                        cv2.line(drawing_canvas, brush_prev_tip, brush_tip, brush_bristle_color, 15, cv2.LINE_AA)
                else:
                    brush_prev_tip = brush_tip
                
                brush_prev_tip = brush_tip
    
    # Display
    cv2.imshow('VTuber Avatar', canvas)
    cv2.resizeWindow('VTuber Avatar', 960, 720)  # Window ukuran 960x720
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()