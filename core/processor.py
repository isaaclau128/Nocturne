import numpy as np

class MusicProcessor:
    def __init__(self):
        # Define the 7 Solfeggio notes (C Major scale as a base)
        self.solfeggio_notes = [60, 62, 64, 65, 67, 69, 71] # MIDI: Do, Re, Mi, Fa, Sol, La, Ti
        self.last_note = None

    def get_dynamics(self, face_landmarks):
        """Calculates volume based on mouth openness (Landmarks 13 & 14)."""
        if not face_landmarks:
            return 0
            
        # Extract Y coordinates for inner lip centers
        upper_lip_y = face_landmarks.landmark[13].y
        lower_lip_y = face_landmarks.landmark[14].y
        
        # Calculate vertical distance (aperture)
        aperture = abs(lower_lip_y - upper_lip_y)
        
        # Map 0.02 - 0.10 aperture to 0 - 127 MIDI velocity
        velocity = np.interp(aperture, [0.01, 0.08], [0, 127])
        return int(np.clip(velocity, 0, 127))

    def get_pitch(self, hand_landmarks, face_landmarks):
        """Calculates MIDI note based on Hand Height + Eye Modifier."""
        if not hand_landmarks:
            return None

        # 1. Base Pitch from Hand Height (Index Finger Tip: Landmark 8)
        # Flip Y so higher hand = higher value (MediaPipe 0.0 is top)
        hand_y = 1.0 - hand_landmarks.landmark[8].y
        
        # Map height to one of the 7 index positions (0-6)
        note_index = int(np.interp(hand_y, [0.2, 0.8], [0, 6.99]))
        base_note = self.solfeggio_notes[note_index]

        # 2. Accidental Modifier from Eye (Iris position)
        # Check if the right iris (Landmark 473) is shifted
        accidental = 0
        if face_landmarks:
            # Simple gaze logic: if iris x is far left/right of eye center
            # For now, let's keep it simple: we can add +1 for sharps later
            pass 

        return base_note + accidental

    def process(self, hand_results, face_results):
        """Combines everything into a single control packet."""
        hand_lm = hand_results.multi_hand_landmarks[0] if hand_results.multi_hand_landmarks else None
        face_lm = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None
        
        pitch = self.get_pitch(hand_lm, face_lm)
        dynamics = self.get_dynamics(face_lm)
        
        return {"pitch": pitch, "velocity": dynamics}
