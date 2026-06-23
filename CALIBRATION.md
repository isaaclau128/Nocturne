# Calibration & Configuration Guide

Fine-tune Nocturne's gesture recognition for your setup and playing style.

## Quick Start

Edit `config.json` and adjust values to match your preferences. Changes take effect immediately on restart.

## Configuration Sections

### Vision Settings

Controls hand and face detection accuracy.

```json
"vision": {
  "hand_detection_confidence": 0.5,      // 0.0-1.0, higher = stricter
  "face_detection_confidence": 0.5,      // 0.0-1.0, higher = stricter
  "max_hands": 1,                        // Currently fixed at 1
  "hand_model_complexity": 0,            // 0-1, higher = slower but more accurate
  "face_refine_landmarks": true          // true for iris tracking
}
```

**When to adjust:**
- **Low hand_detection_confidence** (0.3) — Hand not detected reliably
- **High hand_detection_confidence** (0.8) — Too many false positives
- **Increase model_complexity** — Hand detection unreliable at edges of frame

**Recommended values:**
- Well-lit room: 0.5 (default)
- Dim lighting: 0.3-0.4
- Very bright/outdoor: 0.6-0.7

### Music Settings

Controls gesture-to-MIDI mapping and sensitivity.

```json
"music": {
  "solfeggio_notes": [60, 62, 64, 65, 67, 69, 71],
  "num_notes": 7,
  "mouth_aperture_range": [0.01, 0.08],
  "hand_height_range": [0.2, 0.8],
  "eye_gaze_left_threshold": 0.35,
  "eye_gaze_right_threshold": 0.65,
  "velocity_min": 10,
  "velocity_max": 127
}
```

#### Pitch Mapping

**solfeggio_notes:** MIDI note numbers (middle C = 60)

```
[60, 62, 64, 65, 67, 69, 71]  = C Major (Do, Re, Mi, Fa, Sol, La, Ti)

Change to:
[60, 61, 62, 63, 64, 65, 66]  = Chromatic scale (all semitones)
[60, 62, 64, 67, 69, 72, 74]  = Different voicing
[72, 74, 76, 77, 79, 81, 83]  = One octave higher (add 12 to each)
```

Common scales:
- **C Major:** `[60, 62, 64, 65, 67, 69, 71]`
- **A Minor:** `[57, 59, 60, 62, 64, 65, 67]`
- **Pentatonic:** `[60, 62, 64, 67, 69]` (5 zones instead of 7)

#### Hand Height Mapping

**hand_height_range:** `[low_y, high_y]`

Controls which screen positions map to which notes:
- **low_y = 0.2** — Bottom 20% of frame = Do (lowest note)
- **high_y = 0.8** — Top 80% of frame = Ti (highest note)

**Adjust if:**
- Bottom notes too hard to reach: Lower `low_y` to 0.1
- Top notes hard to reach: Raise `high_y` to 0.9
- Webcam mounted high: Adjust both values downward

#### Mouth Sensitivity

**mouth_aperture_range:** `[min_aperture, max_aperture]`

- **min_aperture = 0.01** — Mouth barely open triggers note
- **max_aperture = 0.08** — Mouth fully open reaches max velocity

**Adjust if:**
- Notes trigger too easily: Raise `min_aperture` to 0.02-0.03
- Hard to reach max volume: Raise `max_aperture` to 0.10-0.12
- Too jerky: Lower both values for smoother dynamics

**Find your range:**
```
1. Close mouth (rest): measure aperture (should be ~0.0)
2. Slightly open: measure aperture (try 0.03)
3. Wide open: measure aperture (max ~0.12)
4. Set range to step 2 → step 3
```

#### Eye Gaze Sensitivity

**eye_gaze_left_threshold** / **eye_gaze_right_threshold**

- **0.35** = Looking noticeably left triggers flat
- **0.65** = Looking noticeably right triggers sharp

Values are iris position relative to eye width:
- **0.0-0.5** = Left side of eye
- **0.5** = Center
- **0.5-1.0** = Right side of eye

**Adjust if:**
- Accidentals trigger too easily: Move thresholds closer (0.4 and 0.6)
- Too hard to trigger: Move apart (0.3 and 0.7)
- Only one direction works: Check lighting affects iris tracking

#### Velocity Thresholds

**velocity_min:** Minimum mouth aperture before note plays
**velocity_max:** Maximum MIDI velocity (standard = 127)

- **velocity_min = 10** — Lowest MIDI velocity for note on
- **velocity_max = 127** — Standard maximum

**Adjust if:**
- Notes play when mouth barely open: Raise `velocity_min` to 20-30
- Want to use full MIDI range: Keep at 127
- Synth too loud: Lower `velocity_max` to 100-110

### MIDI Settings

```json
"midi": {
  "port_name": "GestureSynth",           // Virtual port name (if creating new)
  "note_off_velocity": 0,                // Always 0 (standard)
  "velocity_threshold": 5                // When to turn off note
}
```

**port_name:** Name for virtual MIDI port (macOS/Linux only)
**velocity_threshold:** If mouth opens below this, turns off current note

### Display Settings

```json
"display": {
  "window_title": "Nocturne - Gesture Synth",
  "pitch_zone_color": [255, 255, 255],   // White lines
  "hand_position_color": [255, 0, 0],    // Red circle
  "info_color": [0, 255, 0]              // Green text
}
```

Colors in BGR format (OpenCV convention):
- `[0, 0, 255]` = Red
- `[0, 255, 0]` = Green
- `[255, 0, 0]` = Blue
- `[255, 255, 255]` = White
- `[0, 0, 0]` = Black

### Logging Settings

```json
"logging": {
  "level": "INFO",                       // DEBUG, INFO, WARNING, ERROR
  "log_file": "nocturne.log",
  "max_bytes": 5242880,                  // 5MB
  "backup_count": 3,
  "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
```

**level values:**
- **DEBUG** — Verbose, logs every frame (very chatty)
- **INFO** — Standard, logs initialization and key events
- **WARNING** — Only problems
- **ERROR** — Only failures

## Calibration Workflow

### Step 1: Test Lighting

1. Run `python main.py`
2. Look for red circle (hand detection) and green lines (face zones)
3. If not visible: **Increase lighting**

### Step 2: Calibrate Hand Height

1. Sit naturally with arm at resting position
2. Note the hand's y-position (percentage down screen)
3. Raise arm fully extended
4. Note the hand's y-position
5. Update `hand_height_range` to match:
   ```json
   "hand_height_range": [0.35, 0.15]    // Example: rest at 35%, extended at 15%
   ```

### Step 3: Calibrate Mouth

1. Close mouth completely
2. Open slightly: this is your min aperture
3. Open fully: this is your max aperture
4. Update range:
   ```json
   "mouth_aperture_range": [0.02, 0.10]  // Your measured values
   ```

### Step 4: Calibrate Eyes

1. Look straight at camera
2. Slowly look left until accidental triggers
3. Note how far left you're looking
4. Repeat for right
5. Adjust thresholds:
   ```json
   "eye_gaze_left_threshold": 0.30,   // How far left needed
   "eye_gaze_right_threshold": 0.70   // How far right needed
   ```

### Step 5: Test with DAW

1. Connect synthesizer/DAW
2. Play scales and melodies
3. Adjust sensitivity based on feel
4. Record a simple melody and listen back

## Troubleshooting by Symptom

### Symptom: Notes Play Unexpectedly

**Cause:** Hand or mouth detection too sensitive

**Fix:**
```json
"vision": {"hand_detection_confidence": 0.7},  // More strict
"music": {"mouth_aperture_range": [0.03, 0.10]}  // Requires more mouth
```

### Symptom: Hand Not Detected

**Cause:** Lighting, hand angle, or wrong camera

**Fix:**
1. Improve lighting significantly
2. Wear contrasting color on arm
3. Check camera with:
   ```bash
   python -c "import cv2; cap = cv2.VideoCapture(0); ret, frame = cap.read(); print(f'Frame size: {frame.shape}')"
   ```

### Symptom: Mouth Detection Unreliable

**Cause:** Hair, angle, or lighting

**Fix:**
1. Move hair away from face
2. Face directly at camera (not angled)
3. Improve lighting in face area

### Symptom: Eye Accidentals Don't Work

**Cause:** Iris not tracking (face detection issue) or thresholds wrong

**Fix:**
1. Ensure face detected (green lines visible)
2. Make more extreme eye movements
3. Lower thresholds:
   ```json
   "eye_gaze_left_threshold": 0.25,
   "eye_gaze_right_threshold": 0.75
   ```

### Symptom: Laggy Response

**Cause:** CPU overload or camera latency

**Fix:**
1. Close unnecessary applications
2. Lower model_complexity to 0
3. Reduce camera resolution in system settings
4. Check logs: `grep "WARN" nocturne.log`

### Symptom: Performance Drops After Long Use

**Cause:** Memory leak or log file growth

**Fix:**
1. Increase `backup_count` for log rotation
2. Restart Nocturne periodically
3. Check system memory: `top` or Task Manager

## Performance Optimization

### CPU Usage Reduction

```json
"vision": {
  "hand_model_complexity": 0,           // Faster, less accurate
  "hand_detection_confidence": 0.7      // Skip uncertain detections
},
"logging": {
  "level": "WARNING"                    // Less logging overhead
}
```

### Latency Reduction

1. **Hardware:** USB 3.0 camera preferred
2. **Software:** Close Chrome, Discord, etc.
3. **Config:** Lower detection confidence thresholds
4. **System:** Use Linux for best performance

### Memory Optimization

```json
"logging": {
  "max_bytes": 2097152,                // 2MB instead of 5MB
  "backup_count": 1                    // Keep 1 backup
}
```

## Creating Custom Scales

Want to play in different scales? Modify `solfeggio_notes`:

**Pentatonic (5 zones):**
```json
"solfeggio_notes": [60, 62, 64, 67, 69],
"num_notes": 5
```

**Blues Scale:**
```json
"solfeggio_notes": [60, 63, 65, 66, 67, 70],
"num_notes": 6
```

**Chromatic (all semitones):**
```json
"solfeggio_notes": [60, 61, 62, 63, 64, 65, 66],
"num_notes": 7
```

**Different Key (D Major):**
```json
"solfeggio_notes": [62, 64, 66, 67, 69, 71, 73],  // Add 2 to C Major
"num_notes": 7
```

## Saving Multiple Configurations

Create variant config files:

```bash
cp config.json config_sensitive.json   # High sensitivity version
cp config.json config_live.json        # Performance optimized
cp config.json config_practice.json    # Easier learning settings
```

Load specific config (would need code modification):
```python
# Planned for future: command-line argument
# python main.py --config config_live.json
```

## Testing Changes

After editing `config.json`:

```bash
python -c "import json; json.load(open('config.json')); print('✓ Valid JSON')"
python main.py
```

Invalid JSON will show errors immediately.

## Next Steps

- 🎵 See [USAGE.md](USAGE.md) to learn playing techniques
- 🛠️ Return here if performance changes
- 💾 Save your calibration settings once optimized
- 📊 Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

---

**Fine-tuned and ready to perform! 🎵**
