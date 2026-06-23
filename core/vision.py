import cv2
import mediapipe as mp
from .config import get_config
from .logging_config import get_logger


logger = get_logger(__name__)


class VisionTracker:
    def __init__(self):
        """Initialize vision tracker with config settings."""
        try:
            config = get_config()
            vision_config = config.get_section('vision')
            
            logger.info("Initializing VisionTracker...")
            
            # 1. Initialize MediaPipe Solutions
            self.mp_hands = mp.solutions.hands
            self.mp_face_mesh = mp.solutions.face_mesh
            
            # 2. Setup the actual Detectors with config values
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=vision_config.get('max_hands', 1),
                model_complexity=vision_config.get('hand_model_complexity', 0),
                min_detection_confidence=vision_config.get('hand_detection_confidence', 0.5)
            )
            
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                refine_landmarks=vision_config.get('face_refine_landmarks', True),
                min_detection_confidence=vision_config.get('face_detection_confidence', 0.5)
            )
            
            logger.info("VisionTracker initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VisionTracker: {e}", exc_info=True)
            raise

    def process_frame(self, frame):
        """
        Takes a BGR frame from OpenCV, converts to RGB, 
        and returns the detection results.
        """
        try:
            # Flip frame for "mirror" effect (more natural for music)
            frame = cv2.flip(frame, 1)
            
            # MediaPipe needs RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run the AI models
            hand_results = self.hands.process(rgb_frame)
            face_results = self.face_mesh.process(rgb_frame)
            
            return frame, hand_results, face_results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}", exc_info=True)
            # Return empty results on error
            return frame, None, None

    def release(self):
        """Cleanup resources."""
        try:
            self.hands.close()
            self.face_mesh.close()
            logger.info("VisionTracker resources released")
        except Exception as e:
            logger.error(f"Error releasing VisionTracker resources: {e}", exc_info=True)
