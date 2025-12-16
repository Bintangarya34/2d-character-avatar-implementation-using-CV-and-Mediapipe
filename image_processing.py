
import cv2
import numpy as np
import math

# Orientation buffers untuk hysteresis/debouncing
_body_orientation_buffer = []
_head_orientation_buffer = []
_BUFFER_SIZE = 5  # Butuh 5 frame konsisten sebelum switch

def detect_body_orientation(left_shoulder, right_shoulder, threshold_percent=0.60):
    """
    Deteksi orientasi body (left/right) berdasarkan posisi shoulder.
    Menggunakan hysteresis untuk stabilitas.
    Threshold berbasis PERSENTASE dari shoulder width untuk lebih robust.
    
    Args:
        left_shoulder: Tuple (x, y) posisi left shoulder
        right_shoulder: Tuple (x, y) posisi right shoulder
        threshold_percent: Threshold sebagai persentase dari shoulder width (0.0-1.0)
                         0.60 = 60% dari shoulder width diperlukan untuk trigger
    
    Returns:
        'left' jika body menghadap ke kiri
        'right' jika body menghadap ke kanan
        'center' jika body menghadap ke depan
    """
    global _body_orientation_buffer
    
    if not left_shoulder or not right_shoulder:
        _body_orientation_buffer.clear()
        return 'center'
    
    # Hitung shoulder width (distance antara shoulders)
    shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
    
    # Hitung shoulder difference (positif = right lebih ke kanan)
    shoulder_diff = right_shoulder[0] - left_shoulder[0]
    
    # Threshold berbasis persentase dari shoulder width
    # Gunakan threshold yang lebih besar untuk lebih "lenient" pada center
    effective_threshold = shoulder_width * threshold_percent
    
    # Tentukan orientasi current frame
    if shoulder_diff > effective_threshold:
        current_orientation = 'right'
    elif shoulder_diff < -effective_threshold:
        current_orientation = 'left'
    else:
        current_orientation = 'center'
    
    # Tambah ke buffer
    _body_orientation_buffer.append(current_orientation)
    if len(_body_orientation_buffer) > _BUFFER_SIZE:
        _body_orientation_buffer.pop(0)
    
    # Return majority vote dari buffer
    if len(_body_orientation_buffer) >= _BUFFER_SIZE:
        return max(set(_body_orientation_buffer), key=_body_orientation_buffer.count)
    elif len(_body_orientation_buffer) > 0:
        return _body_orientation_buffer[-1]
    else:
        return 'center'


def detect_head_orientation(left_eye, right_eye, nose, threshold_percent=0.25):
    """
    Deteksi orientasi kepala (left/right tilt) berdasarkan posisi mata dan hidung.
    Menggunakan hysteresis untuk stabilitas.
    Threshold berbasis PERSENTASE dari eye width untuk lebih robust.
    
    Args:
        left_eye: Tuple (x, y) posisi left eye
        right_eye: Tuple (x, y) posisi right eye
        nose: Tuple (x, y) posisi nose
        threshold_percent: Threshold sebagai persentase dari eye width (0.0-1.0)
                         0.15 = 15% dari eye width diperlukan untuk trigger
    
    Returns:
        'left' jika kepala menengok ke kiri
        'right' jika kepala menengok ke kanan
        'center' jika kepala lurus ke depan
    """
    global _head_orientation_buffer
    
    if not left_eye or not right_eye or not nose:
        _head_orientation_buffer.clear()
        return 'center'
    
    # Hitung eye width (distance antara mata)
    eye_width = abs(right_eye[0] - left_eye[0])
    
    # Hitung center mata
    eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
    
    # Hitung nose deviation dari eye center
    nose_diff = nose[0] - eye_center_x
    
    # Threshold berbasis persentase dari eye width
    effective_threshold = eye_width * threshold_percent
    
    # Tentukan orientasi current frame
    if nose_diff > effective_threshold:
        current_orientation = 'right'
    elif nose_diff < -effective_threshold:
        current_orientation = 'left'
    else:
        current_orientation = 'center'
    
    # Tambah ke buffer
    _head_orientation_buffer.append(current_orientation)
    if len(_head_orientation_buffer) > _BUFFER_SIZE:
        _head_orientation_buffer.pop(0)
    
    # Return majority vote dari buffer
    if len(_head_orientation_buffer) >= _BUFFER_SIZE:
        return max(set(_head_orientation_buffer), key=_head_orientation_buffer.count)
    elif len(_head_orientation_buffer) > 0:
        return _head_orientation_buffer[-1]
    else:
        return 'center'

def overlay_rotated_limb(background, img_part, point_a, point_b, scale_width=1.0, angle_offset=90):
    """
    Menempelkan gambar anggota tubuh di antara Titik A dan Titik B.
    Gambar diasumsikan vertikal (berdiri tegak) pada aslinya.
    
    Args:
        background: Canvas BGR image
        img_part: Part image dengan alpha channel (BGRA)
        point_a: Tuple (x, y) untuk starting point
        point_b: Tuple (x, y) untuk ending point
        scale_width: Width scaling multiplier
        angle_offset: Sudut offset rotasi
    
    Returns:
        Canvas dengan limb ter-overlay
    """
    if img_part is None:
        return background
    
    # 1. Hitung Jarak (Euclidean Distance) untuk Scaling Height
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    limb_length = np.sqrt(dx**2 + dy**2)
    
    if limb_length == 0:
        return background

    # Ambil dimensi asli gambar
    h_img, w_img = img_part.shape[:2]
    
    # 2. Scaling
    ratio = w_img / h_img
    new_h = int(limb_length)
    new_w = int(new_h * ratio * scale_width)
    
    if new_w <= 0 or new_h <= 0:
        return background
    
    # Resize dengan INTER_LINEAR untuk speed
    resized_img = cv2.resize(img_part, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 3. Hitung Sudut Rotasi
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) - angle_offset

    # 4. Rotasi Gambar (Affine Transform)
    center = (new_w // 2, new_h // 2)
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)
    
    # Hitung bounding box baru setelah rotasi
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w_rot = int((new_h * sin) + (new_w * cos))
    new_h_rot = int((new_h * cos) + (new_w * sin))
    
    M[0, 2] += (new_w_rot / 2) - center[0]
    M[1, 2] += (new_h_rot / 2) - center[1]
    
    rotated_img = cv2.warpAffine(resized_img, M, (new_w_rot, new_h_rot), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(0,0,0,0))

    # 5. Penempatan (Positioning)
    mid_x = int((point_a[0] + point_b[0]) / 2)
    mid_y = int((point_a[1] + point_b[1]) / 2)
    
    y1, y2 = mid_y - new_h_rot // 2, mid_y + new_h_rot // 2
    x1, x2 = mid_x - new_w_rot // 2, mid_x + new_w_rot // 2
    
    # 6. Alpha Blending
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
    
    # Vectorized Alpha Blending (FAST)
    if overlay_crop.shape[2] == 4:  # Ada alpha channel
        alpha = overlay_crop[:, :, 3:4] / 255.0
        bg_crop = background[y1_clip:y2_clip, x1_clip:x2_clip, :3]
        background[y1_clip:y2_clip, x1_clip:x2_clip, :3] = (
            alpha * overlay_crop[:, :, :3] + (1.0 - alpha) * bg_crop
        ).astype(np.uint8)
    else:
        background[y1_clip:y2_clip, x1_clip:x2_clip] = overlay_crop
                                                           
    return background


def draw_head(canvas, head_img, nose_pos, left_eye_pos, right_eye_pos, body_scale, head_scale_mult, position_offset):
    """
    Draw kepala dan mata (dengan blinking)
    
    Args:
        canvas: BGR image canvas
        head_img: Head image dengan alpha
        nose_pos: Position mata
        left_eye_pos: Left eye position
        right_eye_pos: Right eye position
        body_scale: Scale untuk kepala
        head_scale_mult: Head multiplier
        position_offset: Y offset untuk posisi kepala
    """
    if head_img is None:
        return canvas
    
    h_img, w_img = head_img.shape[:2]
    new_w = int(w_img * body_scale * head_scale_mult)
    new_h = int(h_img * body_scale * head_scale_mult)
    
    if new_w <= 0 or new_h <= 0:
        return canvas
    
    resized_head = cv2.resize(head_img, (new_w, new_h))
    
    # Posisi kepala di atas nose
    y1 = int(nose_pos[1] - new_h * position_offset)
    x1 = int(nose_pos[0] - new_w // 2)
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
                    canvas[y1_clip:y2_clip, x1_clip:x2_clip, c] = (
                        alpha_s[:, :, 0] * overlay_crop[:, :, c] + 
                        (1.0 - alpha_s[:, :, 0]) * canvas[y1_clip:y2_clip, x1_clip:x2_clip, c]
                    ).astype(np.uint8)
    
    return canvas
