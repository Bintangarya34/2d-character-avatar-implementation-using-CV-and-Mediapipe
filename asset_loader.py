"""
Asset Loading & Management
"""
import cv2
import numpy as np
import os
from config import assets

def load_assets():
    """Load semua asset images dengan alpha channel"""
    images = {}
    
    print("Loading Assets...")
    for key, filename in assets.items():
        filepath = os.path.join('img', filename)
        if os.path.exists(filepath):
            # IMREAD_UNCHANGED untuk transparansi
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Error: Gagal load {filepath}")
            else:
                # Tidak perlu resize/optimisasi asset, langsung pakai gambar asli
                
                images[key] = img
                print(f"Loaded {filepath} (size: {img.shape})")
                
                # Mirror gambar lengan kiri dari kanan
                if 'left_arm' in key:
                    images[key] = cv2.flip(img, 1)  # Flip horizontal
        else:
            print(f"Warning: File {filepath} tidak ditemukan!")
    
    return images


def generate_eye_image(eye_size=30):
    """Generate simple eye image (black circle dengan alpha)"""
    eye_img = np.zeros((eye_size, eye_size, 4), dtype=np.uint8)
    cv2.circle(eye_img, (eye_size//2, eye_size//2), eye_size//2 - 2, (0, 0, 0, 255), -1)
    return eye_img
