"""
Utility Functions - Helper functions
"""
import numpy as np




def calculate_body_scale(shoulder_left, shoulder_right, reference_distance, scale_min, scale_max):
    """
    Hitung body scale dari shoulder distance
    
    Args:
        shoulder_left: Left shoulder position
        shoulder_right: Right shoulder position
        reference_distance: Reference distance untuk scale
        scale_min: Minimum scale
        scale_max: Maximum scale
    
    Returns:
        Clamped body scale
    """
    if not shoulder_left or not shoulder_right:
        return 1.0
    
    shoulder_distance = np.sqrt(
        (shoulder_right[0] - shoulder_left[0])**2 + 
        (shoulder_right[1] - shoulder_left[1])**2
    )
    
    body_scale = shoulder_distance / reference_distance
    return np.clip(body_scale, scale_min, scale_max)


def get_point(landmark_idx, landmarks, w, h, landmark_buffer, smoothing_factor):
    # Return raw landmark position (tanpa smoothing)
    if landmark_idx < len(landmarks) and landmarks[landmark_idx].visibility > 0.5:
        return (int(landmarks[landmark_idx].x * w), int(landmarks[landmark_idx].y * h))
    return None


def detect_user_blinking(landmarks, blink_threshold=0.25):
    """
    Deteksi apakah user sedang merem/kedip dengan membandingkan jarak vertikal dan horizontal mata.
    
    Menggunakan simple method: jika vertical distance mata < horizontal distance * threshold
    berarti mata sedang tertutup (merem)
    
    Args:
        landmarks: MediaPipe pose landmarks (468 points)
        blink_threshold: Ratio threshold (0.25 = jika vertical < 25% dari horizontal, dianggap merem)
    
    Returns:
        True jika user sedang merem/kedip, False jika mata terbuka
    """
    try:
        # Eye point indices dari MediaPipe (dari face mesh)
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 362, 385, 387, 263, 373, 380
        
        # Simplified: gunakan top dan bottom eyelid
        left_eye_top = landmarks[159]  # Top eyelid left
        left_eye_bottom = landmarks[145]  # Bottom eyelid left
        left_eye_left = landmarks[33]  # Left corner
        left_eye_right = landmarks[133]  # Right corner
        
        right_eye_top = landmarks[386]  # Top eyelid right
        right_eye_bottom = landmarks[374]  # Bottom eyelid right
        right_eye_left = landmarks[362]  # Left corner
        right_eye_right = landmarks[263]  # Right corner
        
        # Calculate vertical distances (mata terbuka/tertutup)
        left_vert = abs(left_eye_top.y - left_eye_bottom.y)
        right_vert = abs(right_eye_top.y - right_eye_bottom.y)
        
        # Calculate horizontal distances (width of eye)
        left_horiz = abs(left_eye_right.x - left_eye_left.x)
        right_horiz = abs(right_eye_right.x - right_eye_left.x)
        
        # Average ratios
        left_ratio = left_vert / left_horiz if left_horiz > 0 else 0.5
        right_ratio = right_vert / right_horiz if right_horiz > 0 else 0.5
        avg_ratio = (left_ratio + right_ratio) / 2.0
        
        # Jika ratio kecil (< threshold), berarti mata tertutup
        is_blinking = avg_ratio < blink_threshold
        
        return is_blinking
    except:
        # Fallback jika ada error
        return False
