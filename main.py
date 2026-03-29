import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two landmark points."""
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

def main():
    # 1. Setup Webcam
    cap = cv2.VideoCapture(0)
    
    # 2. Initialize Models
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
        
        print("AI Instrument Running... Press 'q' to quit.")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            # Flip and convert for MediaPipe
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process Frame
            hand_results = hands.process(rgb_frame)
            face_results = face_mesh.process(rgb_frame)

            # --- LOGIC: MOUTH (DYNAMICS) ---
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Landmarks 13 & 14 are inner lip centers
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]
                    
                    mouth_opening = calculate_distance(upper_lip, lower_lip)
                    
                    # Visual Feedback for Dynamics
                    cv2.putText(frame, f"Dynamics (Mouth): {round(mouth_opening, 3)}", 
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # --- LOGIC: HAND (SOLFEGGIO) ---
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Index Finger Tip (Landmark 8)
                    index_tip = hand_landmarks.landmark[8]
                    
                    # Visual Feedback for Pitch
                    h, w, _ = frame.shape
                    for i in range(7):
                        # Draw horizontal lines for the 7 zones
                        y_line = int(h * (0.2 + (i * 0.085))) # Matches the 0.2-0.8 mapping in processor
                        cv2.line(frame, (0, y_line), (w, y_line), (255, 255, 255), 1)
                        cv2.putText(frame, f"Note {i}", (10, y_line - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                    cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
                    
                    # TODO: Map 'cy' (height) to Solfeggio notes
                    cv2.putText(frame, f"Hand Y: {round(index_tip.y, 2)}", 
                                (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display Window
            cv2.imshow('Gesture Synth Prototype', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


