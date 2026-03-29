import cv2
import mediapipe as mp

class VisionTracker:
    def __init__(self):
        # 1. Initialize MediaPipe Solutions
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # 2. Setup the actual Detectors
        # we use low complexity for faster real-time performance
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True, # Critical for Iris tracking
            min_detection_confidence=0.5
        )

    def process_frame(self, frame):
        """
        Takes a BGR frame from OpenCV, converts to RGB, 
        and returns the detection results.
        """
        # Flip frame for "mirror" effect (more natural for music)
        frame = cv2.flip(frame, 1)
        
        # MediaPipe needs RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run the AI models
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        return frame, hand_results, face_results

    def release(self):
        """Cleanup resources."""
        self.hands.close()
        self.face_mesh.close()
