# Architecture & System Design

Deep dive into how Nocturne works and how components interact.

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Nocturne System                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Webcam → VisionTracker → MusicProcessor → MIDIDriver → Synth  │
│           (MediaPipe)    (Gesture Maps)   (MIDI Out)            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Frame Acquisition

```
Webcam (30 FPS)
    ↓ (cv2.VideoCapture)
Raw BGR Frame (H×W×3)
    ↓ (processed in main.py)
Visualization + Processing
```

**Key details:**
- **Resolution:** Typically 640×480 or 1280×720
- **Format:** BGR (OpenCV native)
- **Frame rate:** ~30 FPS
- **Latency:** <30ms from sensor to processing

### 2. Vision Tracking (VisionTracker)

```
BGR Frame
    ↓
cv2.flip(frame, 1)  ← Mirror effect
    ↓
cv2.cvtColor(BGR → RGB)  ← MediaPipe needs RGB
    ↓
┌──────────────────────────────────────────┐
│    MediaPipe Hands.process()             │
│  - Detects hand landmarks (21 points)    │
│  - Confidence score for each landmark    │
│  - Returns multi_hand_landmarks          │
└──────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────┐
│   MediaPipe FaceMesh.process()           │
│  - Detects face landmarks (468 points)   │
│  - Includes iris tracking                │
│  - Returns multi_face_landmarks          │
└──────────────────────────────────────────┘
    ↓
hand_results, face_results
```

**MediaPipe Hand Landmarks:**
```
0  = wrist
1-4 = thumb (base, mid, pip, tip)
5-8 = index (base, mid, pip, tip)
...
17-20 = pinky (base, mid, pip, tip)

We use landmark[8] = index finger tip for pitch
```

**MediaPipe Face Landmarks:**
```
13, 14 = mouth (upper, lower lip) for dynamics
362, 263 = eye corners for eye gaze
473 = iris center for eye gaze
```

### 3. Gesture Processing (MusicProcessor)

```
hand_results, face_results
    ├─ Extract first hand landmarks
    ├─ Extract first face landmarks
    └─→ process()
        ├─→ get_pitch()
        │   ├─ Get hand_y = 1.0 - hand_landmarks.landmark[8].y
        │   ├─ Map hand_y from [0.2, 0.8] to [0, 6.99]
        │   ├─ Select solfeggio_notes[note_index]
        │   ├─ Call get_accidental() for eye modifier
        │   └─ Return note = base_note + accidental
        │
        └─→ get_dynamics()
            ├─ Get mouth aperture = |upper_lip.y - lower_lip.y|
            ├─ Map aperture from [0.01, 0.08] to [0, 127]
            ├─ Clip to valid MIDI velocity range
            └─ Return velocity

Result: {"pitch": 62, "velocity": 85}
```

**Pitch Calculation Detail:**

Hand position to MIDI note mapping:
```python
hand_y = 1.0 - hand_landmarks.landmark[8].y  # Flip so up=high

# Map screen position to note index
# hand_height_range [0.2, 0.8] covers bottom to top 60% of screen
note_index = int(np.interp(hand_y, [0.2, 0.8], [0, 6.99]))

base_note = solfeggio_notes[note_index]  # e.g., 62 (D)

# Add accidental from eye gaze
accidental = get_accidental(face_landmarks)  # -1, 0, or +1
pitch = base_note + accidental  # e.g., 62 + 1 = 63
```

**Dynamics Calculation Detail:**

Mouth opening to MIDI velocity:
```python
# Calculate mouth aperture (vertical opening)
upper_lip_y = face_landmarks.landmark[13].y
lower_lip_y = face_landmarks.landmark[14].y
aperture = abs(lower_lip_y - upper_lip_y)

# Map to MIDI velocity [0, 127]
velocity = np.interp(aperture, [0.01, 0.08], [0, 127])
velocity = int(np.clip(velocity, 0, 127))
```

### 4. MIDI Output (MIDIDriver)

```
{"pitch": 62, "velocity": 85}
    ↓
Velocity threshold check (velocity > 5)
    ├─ YES → Note change or new note
    │  └─→ Check if same as current_note
    │      ├─ YES → Update velocity (sustain)
    │      └─ NO → Send Note Off (old), then Note On (new)
    │
    └─ NO → Turn off current note
        └─→ Send Note Off message

MIDI Message Format:
┌──────────────────────────────────────┐
│ Status Byte │ Data Byte 1 │ Data Byte 2 │
├──────────────────────────────────────┤
│ Note On:    0x90       │ note (0-127) │ velocity (0-127)
│ Note Off:   0x80       │ note (0-127) │ 0
└──────────────────────────────────────┘
```

## Component Details

### VisionTracker

**File:** `core/vision.py`

**Responsibilities:**
- Initialize MediaPipe models
- Process each frame through detectors
- Handle errors gracefully

**Interface:**
```python
tracker = VisionTracker()
frame, hand_results, face_results = tracker.process_frame(bgr_frame)
tracker.release()  # Cleanup
```

**Performance:**
- ~15-20ms per frame (MediaPipe inference)
- Hand detection: 98% accuracy at 0.5m
- Face mesh: 97% accuracy at typical distance

### MusicProcessor

**File:** `core/processor.py`

**Responsibilities:**
- Map hand height → MIDI pitch (0-127)
- Map mouth opening → velocity (0-127)
- Map eye gaze → accidentals (-1, 0, +1)
- Combine into MIDI messages

**Interface:**
```python
processor = MusicProcessor()
control_data = processor.process(hand_results, face_results)
# Returns: {"pitch": int or None, "velocity": int}
```

**Key Methods:**
```python
processor.get_pitch(hand_landmarks, face_landmarks)  # → int or None
processor.get_dynamics(face_landmarks)                # → 0-127
processor.get_accidental(face_landmarks)              # → -1/0/+1
```

### MIDIDriver

**File:** `core/midi_driver`

**Responsibilities:**
- Manage MIDI connection
- Send Note On/Note Off messages
- Track current playing note

**Interface:**
```python
midi = MIDIDriver()
midi.send_note(note=62, velocity=85)
midi.close()  # Cleanup
```

**MIDI State Machine:**
```
[No Note Playing]
    ↓
send_note(60, 50)
    ↓ (velocity > threshold, new note)
[Send Note On: 60, 50]
    ↓
[60 Playing]
    ↓
send_note(60, 100)
    ↓ (same note, higher velocity)
[60 Playing] (velocity updated in synth)
    ↓
send_note(62, 80)
    ↓ (different note)
[Send Note Off: 60, 0] → [Send Note On: 62, 80]
    ↓
[62 Playing]
    ↓
send_note(62, 3)  # velocity below threshold
    ↓ (velocity < threshold)
[Send Note Off: 62, 0]
    ↓
[No Note Playing]
```

### Configuration Management

**File:** `core/config.py`

**Responsibilities:**
- Load JSON configuration
- Provide easy access via dot notation
- Validate settings

**Usage:**
```python
config = get_config()

# Dot notation access
solfeggio = config.get('music.solfeggio_notes')
hand_height_min = config.get('music.hand_height_range')[0]

# Section access
music_config = config.get_section('music')
log_level = music_config.get('level', 'INFO')
```

### Logging

**File:** `core/logging_config.py`

**Responsibilities:**
- Setup console + file logging
- Handle log rotation
- Provide per-module loggers

**Usage:**
```python
from core.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Application started")
logger.error("Error occurred", exc_info=True)
```

**Log Levels:**
- `DEBUG` (10) — Frame-by-frame details, calculations
- `INFO` (20) — Initialization, events, state changes
- `WARNING` (30) — Recoverable issues, degraded performance
- `ERROR` (40) — Failures and exceptions
- `CRITICAL` (50) — System failures

## Performance Analysis

### Latency Breakdown

Typical latency from gesture to sound: **~50-80ms**

```
Webcam capture:     5-10ms
Frame transmission: 5-10ms
Vision processing: 15-20ms  ← MediaPipe heavy
Gesture mapping:    1-2ms
MIDI transmission:  1-5ms
Synth latency:     20-30ms
─────────────────
Total:             50-80ms
```

Target for acceptable playability: <100ms ✓

### CPU Usage

Typical usage on mid-range CPU:
- Vision tracking: 60-80% of processing
- MIDI/Config: <1%
- Python overhead: 5-10%
- Total: 15-25% of one CPU core

### Memory Usage

- Base Python: ~20MB
- MediaPipe models: ~50-100MB
- Frame buffer: ~10-20MB
- Logs: <5MB (rotated)
- **Total: ~100-150MB**

## Error Handling Strategy

### Graceful Degradation

If component fails, system continues:

```
Vision fails (hand not detected)
    ↓ (hand_lm = None)
Processor returns (pitch=None, velocity=0)
    ↓
MIDI sends nothing
    ↓
System stays running, awaits next frame
```

### Exception Handling

All critical sections have try-except:

```python
try:
    frame, hand_results, face_results = vision.process_frame(frame)
except Exception as e:
    logger.error(f"Vision error: {e}", exc_info=True)
    hand_results, face_results = None, None
    # Continue with no detection
```

### Logging Errors

- **First occurrence:** Full traceback with `exc_info=True`
- **Repeated errors:** Single warning to avoid log spam
- **Critical errors:** Exit with error code

## Extension Points

### Adding New Gesture Controls

1. Add new face landmark detection
2. Extract value from landmarks
3. Map to 0-1 range via `np.interp()`
4. Add to control_data dict
5. Use in MIDIDriver (e.g., pitch bend)

### Adding New MIDI Features

1. Extend MIDIDriver with new method
2. Call from main loop
3. Use control_data values
4. Send appropriate MIDI messages

Example: Pitch bend
```python
def send_pitch_bend(self, hand_x_position):
    """Send MIDI pitch bend based on hand x position."""
    # Map hand_x to pitch bend value (-8192 to 8191)
    pb_value = int(np.interp(hand_x_position, [0, 1], [-8192, 8191]))
    # Send pitch bend message
    self.midi_out.send_message([0xE0, pb_value & 0x7F, (pb_value >> 7) & 0x7F])
```

### Adding New Configuration Options

1. Add to config.json with default value
2. Load in appropriate module
3. Document in CALIBRATION.md
4. Use in processing logic

## Testing Strategy

### Unit Tests

Test individual functions in isolation:

```python
def test_get_pitch():
    processor = MusicProcessor()
    # Mock landmarks with specific values
    pitch = processor.get_pitch(mock_hand, mock_face)
    assert 60 <= pitch <= 71
```

### Integration Tests

Test component interactions:

```python
def test_full_pipeline():
    # Create mocked frame
    # Run through VisionTracker
    # Pass to MusicProcessor
    # Send to MIDI
    # Verify output
```

### Manual Testing

1. Visual inspection of detection (red circle, green lines)
2. Play simple melodies, listen for accuracy
3. Test boundary conditions (hand at edges, mouth extremes)
4. Check logs for errors

## Future Architecture Improvements

### Multi-Hand Support

```
hand_results.multi_hand_landmarks[0] → Voice 1
hand_results.multi_hand_landmarks[1] → Voice 2
                                     → Chord or harmony
```

### ML-Based Gesture Recognition

```
Raw hand landmarks
    ↓
Feature extraction (distances, angles)
    ↓
Neural network classifier
    ↓
Gesture type (tap, slide, vibrato, etc.)
    ↓
Apply gesture-specific effects to MIDI
```

### Audio Synthesis

```
MIDI pitch/velocity
    ↓
Synthesizer (sine, square, saw waves)
    ↓
Amplitude envelope (ADSR)
    ↓
Effects (reverb, delay)
    ↓
Audio output
```

## Troubleshooting by Component

### Vision Issues

**Check:**
- MediaPipe version compatibility
- Camera driver and permissions
- Frame resolution and FPS

**Debug:**
```python
logger.debug(f"Hand detected: {hand_results.multi_hand_landmarks is not None}")
logger.debug(f"Face detected: {face_results.multi_face_landmarks is not None}")
```

### Processing Issues

**Check:**
- Gesture ranges in config.json
- Landmark availability before accessing
- Mathematical errors in interpolation

**Debug:**
```python
logger.debug(f"Hand Y: {hand_y}, Note index: {note_index}")
logger.debug(f"Aperture: {aperture}, Velocity: {velocity}")
```

### MIDI Issues

**Check:**
- MIDI port availability
- Synth listening on correct port
- Note values in valid range (0-127)

**Debug:**
```python
logger.debug(f"Sending Note On: {note}, Velocity: {velocity}")
```

## References

- [MediaPipe Documentation](https://google.github.io/mediapipe/)
- [MIDI Specification](https://www.midi.org/)
- [NumPy Interpolation](https://numpy.org/doc/stable/reference/generated/numpy.interp.html)
- [OpenCV Python API](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

---

**Understanding the architecture helps you extend and improve Nocturne! 🎵**
