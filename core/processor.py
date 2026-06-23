import numpy as np
import os
from collections import deque
from .config import get_config
from .logging_config import get_logger


logger = get_logger(__name__)


try:
    from .recognizer import SignRecognizer
except Exception:
    SignRecognizer = None


class MusicProcessor:
    def __init__(self):
        """Initialize music processor with config settings and optional recognizer."""
        try:
            config = get_config()
            self.music_config = config.get_section('music')
            self.midi_config = config.get_section('midi')
            self.recog_config = config.get_section('recognizer')

            logger.info("Initializing MusicProcessor...")

            # Load solfeggio notes from config
            self.solfeggio_notes = self.music_config.get('solfeggio_notes', [60, 62, 64, 65, 67, 69, 71])
            self.last_note = None

            # Extract config values
            self.aperture_min, self.aperture_max = self.music_config.get('mouth_aperture_range', [0.01, 0.08])
            self.hand_height_min, self.hand_height_max = self.music_config.get('hand_height_range', [0.2, 0.8])
            self.eye_gaze_left = self.music_config.get('eye_gaze_left_threshold', 0.35)
            self.eye_gaze_right = self.music_config.get('eye_gaze_right_threshold', 0.65)
            self.velocity_min = self.music_config.get('velocity_min', 10)
            self.velocity_max = self.music_config.get('velocity_max', 127)

            # Recognizer config
            self.recognizer_enabled = bool(self.recog_config.get('enabled', False))
            self.recognizer_path = self.recog_config.get('model_path')
            self.debounce_frames = int(self.recog_config.get('debounce_frames', 3))
            self.label_to_note = self.recog_config.get('label_to_note', {})
            self.pred_history = deque(maxlen=self.debounce_frames)
            self.stable_label = None

            self.recognizer = None
            if self.recognizer_enabled and SignRecognizer and self.recognizer_path:
                try:
                    self.recognizer = SignRecognizer(self.recognizer_path)
                    if not self.recognizer.is_loaded():
                        logger.warning(f"Recognizer model not found or failed to load: {self.recognizer_path}")
                        self.recognizer = None
                    else:
                        logger.info(f"Loaded recognizer model from {self.recognizer_path}")
                except Exception as e:
                    logger.warning(f"Error loading recognizer: {e}")
                    self.recognizer = None

            logger.info(f"MusicProcessor initialized with {len(self.solfeggio_notes)} notes: {self.solfeggio_notes}")

        except Exception as e:
            logger.error(f"Error initializing MusicProcessor: {e}", exc_info=True)
            raise

    def get_dynamics(self, face_landmarks):
        """Calculates volume based on mouth openness (Landmarks 13 & 14)."""
        try:
            if not face_landmarks:
                return 0
                
            # Extract Y coordinates for inner lip centers
            upper_lip_y = face_landmarks.landmark[13].y
            lower_lip_y = face_landmarks.landmark[14].y
            
            # Calculate vertical distance (aperture)
            aperture = abs(lower_lip_y - upper_lip_y)
            
            # Map aperture range to velocity range
            velocity = np.interp(aperture, [self.aperture_min, self.aperture_max], [0, self.velocity_max])
            return int(np.clip(velocity, 0, self.velocity_max))
            
        except Exception as e:
            logger.warning(f"Error calculating dynamics: {e}")
            return 0

    def get_pitch(self, hand_landmarks, face_landmarks):
        """Calculates MIDI note based on Hand Height + Eye Modifier."""
        try:
            if not hand_landmarks:
                return None

            # 1. Base Pitch from Hand Height (Index Finger Tip: Landmark 8)
            # Flip Y so higher hand = higher value (MediaPipe 0.0 is top)
            hand_y = 1.0 - hand_landmarks.landmark[8].y
            
            # Map height to one of the note index positions
            num_notes = len(self.solfeggio_notes)
            max_index = num_notes - 0.01
            note_index = int(np.interp(hand_y, [self.hand_height_min, self.hand_height_max], [0, max_index]))
            base_note = self.solfeggio_notes[note_index]

            # 2. Accidental Modifier from Eye (Iris position)
            accidental = self.get_accidental(face_landmarks)

            return base_note + accidental
            
        except Exception as e:
            logger.warning(f"Error calculating pitch: {e}")
            return None

    def process(self, hand_results, face_results):
        """Combines everything into a single control packet."""
        try:
            hand_lm = hand_results.multi_hand_landmarks[0] if hand_results and hand_results.multi_hand_landmarks else None
            face_lm = face_results.multi_face_landmarks[0] if face_results and face_results.multi_face_landmarks else None

            # If recognizer available, try to detect explicit solfege signs
            pitch = None
            if self.recognizer is not None and hand_lm is not None:
                label = self.recognizer.predict(hand_lm.landmark)
                self.pred_history.append(label)
                # check for stable label across history
                if len(self.pred_history) == self.pred_history.maxlen and all(p == self.pred_history[0] for p in self.pred_history):
                    stable = self.pred_history[0]
                    if stable and stable in self.label_to_note:
                        pitch = int(self.label_to_note[stable])
                        self.stable_label = stable
                # if no stable label, do not change pitch (fallback below)

            # Fallback: map by hand height if no recognizer pitch
            if pitch is None:
                pitch = self.get_pitch(hand_lm, face_lm)

            dynamics = self.get_dynamics(face_lm)

            return {"pitch": pitch, "velocity": dynamics}
            
        except Exception as e:
            logger.error(f"Error in process: {e}", exc_info=True)
            return {"pitch": None, "velocity": 0}

    def get_accidental(self, face_landmarks):
        """Detects if the user is looking left (Flat) or right (Sharp)."""
        try:
            if not face_landmarks:
                return 0

            # Right Eye Landmarks: 473 (Iris Center), 362 (Inner Corner), 263 (Outer Corner)
            iris = face_landmarks.landmark[473].x
            inner_corner = face_landmarks.landmark[362].x
            outer_corner = face_landmarks.landmark[263].x

            # Calculate relative position (0.0 to 1.0)
            total_width = abs(outer_corner - inner_corner)
            if total_width < 0.001:  # Avoid division by very small numbers
                return 0
                
            relative_pos = (iris - inner_corner) / total_width

            if relative_pos < self.eye_gaze_left:  # Looking noticeably Left
                return -1  # Flat
            elif relative_pos > self.eye_gaze_right:  # Looking noticeably Right
                return 1  # Sharp
            
            return 0  # Natural
            
        except Exception as e:
            logger.debug(f"Error calculating accidental: {e}")
            return 0
