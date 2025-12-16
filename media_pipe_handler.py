"""
MediaPipe Handler - Semua setting & handling MediaPipe di sini
"""
import mediapipe as mp
import config

class PoseHandler:
    """Handler untuk pose detection"""
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            model_complexity=config.POSE_MODEL_COMPLEXITY
        )
    
    def process(self, rgb_frame):
        """Process frame dan return hasil"""
        return self.pose.process(rgb_frame)
    
    def close(self):
        """Close pose detector"""
        self.pose.close()


class HandsHandler:
    """Handler untuk hand detection"""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.HANDS_MAX_NUM,
            min_detection_confidence=config.HANDS_MIN_DETECTION_CONFIDENCE,
            model_complexity=config.HANDS_MODEL_COMPLEXITY
        )
    
    def process(self, rgb_frame):
        """Process frame dan return hasil"""
        return self.hands.process(rgb_frame)
    
    def close(self):
        """Close hand detector"""
        self.hands.close()
