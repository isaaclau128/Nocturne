# Usage Guide

How to play Nocturne and master gesture-based music synthesis.

## Getting Started

### Before Playing

1. Ensure good lighting (natural or artificial)
2. Position webcam at eye level, ~2 feet away
3. Wear contrasting clothing (helps hand detection)
4. Connect MIDI synthesizer or DAW
5. Start Nocturne: `python main.py`

### What You'll See

The application shows:
- **Horizontal pitch lines** — 7 zones for Do, Re, Mi, Fa, Sol, La, Ti
- **Red circle** — Your index finger position
- **Green text** — Current velocity (dynamics)
- **Crosshairs** — Reference lines for zones

## Playing Basics

### 1. Hand Position → Pitch

Your hand height controls which of the 7 notes plays:

```
↑ HIGH  (Top screen)     = Ti (B)
        = La (A)
        = Sol (G)
        = Fa (F)
        = Mi (E)
        = Re (D)
↓ LOW   (Bottom screen)  = Do (C)
```

**Technique:**
- Keep your hand in front of camera
- Move hand up/down to change pitch
- Higher hand = higher note
- Index finger (tip) determines position

### 2. Mouth Opening → Dynamics (Volume)

Mouth aperture controls velocity (0-127 MIDI):

- **Mouth closed** → No sound (velocity ~0)
- **Mouth slightly open** → Soft (velocity 30-60)
- **Mouth wide open** → Loud (velocity 100-127)

**Technique:**
- Open naturally, don't force
- Smooth opening/closing = smooth volume curves
- Good for crescendos and dynamics expression

### 3. Eye Gaze → Accidentals (Sharps/Flats)

Where you look modifies the note by a semitone:

- **Looking left** → Flat (-1 semitone)
- **Looking center** → Natural (0)
- **Looking right** → Sharp (+1 semitone)

**Technique:**
- Subtle eye movements (don't need extreme gazes)
- Combine with hand position for full chromatic scale
- Example: Hand at "Do" + look right = Do# (C#)

## Playing Techniques

### Single Notes

1. Position hand at desired height
2. Open mouth to trigger note
3. Close mouth to stop
4. Hand stays still while you sustain

### Melodies

1. Keep mouth open at consistent level
2. Move hand up/down smoothly to play melody
3. Varies pitch while sustaining (no re-triggering)
4. Smooth hand movements = legato effect

### Slides

1. Hold note with mouth open
2. Smoothly raise/lower hand
3. Creates pitch glide effect (portamento)

### Staccato

1. Quick hand jerks while mouth opens/closes rapidly
2. Each mouth movement = separate note
3. Creates punchy, articulate sound

### Chords/Harmony

**Note:** Currently limited to single notes. Future versions will support chords via multiple hands or multi-touch.

**Workaround:**
- Use a DAW or synth with arpeggiator
- Record individual note sequences
- Layer multiple Nocturne outputs

### Expression Techniques

| Technique | How | Effect |
|-----------|-----|--------|
| **Vibrato** | Slight mouth pulsing | Adds warmth/vibration |
| **Dynamics** | Vary mouth opening | Dynamic swells |
| **Trill** | Rapid hand movement | Rapid note repetition |
| **Bend** | Slow hand movement | Pitch gliding |
| **Accent** | Quick mouth open/close | Percussive attacks |

## Configuration for Playing

Edit `config.json` to customize feel:

```json
{
  "music": {
    "mouth_aperture_range": [0.01, 0.08],    // Lower = more sensitive
    "hand_height_range": [0.2, 0.8],         // Adjust for your webcam angle
    "eye_gaze_left_threshold": 0.35,         // How far left to trigger flat
    "eye_gaze_right_threshold": 0.65,        // How far right to trigger sharp
    "velocity_min": 10,                      // Minimum velocity threshold
    "velocity_max": 127                      // Maximum velocity
  }
}
```

See [CALIBRATION.md](CALIBRATION.md) for detailed tuning.

## Tips & Tricks

### Lighting

- **Best:** Natural window light + no glare
- **Good:** Bright overhead lighting
- **Avoid:** Backlighting, shadows on hands
- **Problem:** Adjusts camera brightness in settings

### Hand Position

- **Keep steady:** Reduces jitter
- **Arm supported:** Elbow on table reduces fatigue
- **Relax:** Tension = shaky notes
- **Distance:** 2-3 feet from camera works best

### Mouth Movements

- **Clear view:** Don't cover mouth with hair/hand
- **Avoid fast:** Smooth opening is more expressive than jerky
- **Consistency:** Practice at same mouth speed for predictable dynamics

### Eyes

- **Subtle:** Don't need extreme gaze, 30-45° off-center works
- **Smooth:** Gradual eye movement = gradual pitch changes
- **Natural:** Blink normally; won't interfere

### Performance

- **Warm up:** A few scale runs before serious playing
- **Practice:** Start with simple melodies
- **Record:** Enable DAW recording to capture performances
- **Iterate:** Re-record sections you want to improve

## Common Issues While Playing

### Hand Not Detected

1. **Cause:** Poor lighting or hand too far away
2. **Fix:** Improve lighting, move closer to camera, wear contrasting sleeve

### Mouth Not Detected

1. **Cause:** Face not clearly visible, hair covering face
2. **Fix:** Move hair aside, better lighting, closer to camera

### Notes Randomly Triggering

1. **Cause:** Sensitivity too high or accidental detection
2. **Fix:** Lower `velocity_min` or `mouth_aperture_range` in config
3. **See:** [CALIBRATION.md](CALIBRATION.md)

### Laggy/Delayed Response

1. **Cause:** High CPU load or webcam bottleneck
2. **Fix:** Check `nocturne.log` for warnings, reduce other programs
3. **Improve:** Better webcam (USB 3.0 recommended)

### MIDI Not Sending

1. **Cause:** DAW/synth not listening or port not connected
2. **Check:** `python -c "import rtmidi; m = rtmidi.MidiOut(); print(m.get_ports())"`
3. **Fix:** Restart DAW/synth, reconnect USB, restart Nocturne

## Practicing Exercises

### Scale Run

Play Do-Re-Mi-Fa-Sol-La-Ti-Do (C major scale):
1. Start hand at bottom (Do)
2. Smoothly raise to each line
3. Keep mouth consistently open
4. Should take 2-3 seconds

### Note Accuracy

1. Place hand at middle zone (Mi/E)
2. Close mouth to stop note
3. Open mouth again to retrigger at same pitch
4. Repeat 5 times
5. Listen for consistency

### Dynamics Control

1. Play single note (hand at Mi)
2. Gradually open mouth from closed to wide over 2 seconds
3. Should hear smooth crescendo
4. Reverse: close mouth smoothly for decrescendo

### Eye Control

1. Play note at middle pitch (Do at center)
2. Look slowly left (triggers flat)
3. Look back to center (natural)
4. Look right (triggers sharp)
5. Should hear 3 different pitches seamlessly

### Full Melody

Try these simple melodies to practice all controls:

**Happy Birthday:**
```
Do Do Mi Do Fa Mi
(move hand, vary mouth for dynamics)
```

**Twinkle Twinkle Little Star:**
```
Do Do Sol Sol La La Sol Fa Fa Mi Mi Re Re Do
```

**Simple Song:**
```
Mi Mi Mi | Fa Sol | La La La
```

## Connecting to DAW

### Ableton Live

1. Nocturne → Preferences → MIDI Ports
2. Enable "Nocturne" input port
3. Create MIDI track
4. Arm track for recording
5. Play and Nocturne sends MIDI

### FL Studio

1. Click MIDI Settings
2. Scroll to Input Devices
3. Select Nocturne port
4. Assign to instrument track
5. Start playing

### Logic Pro

1. Preferences → MIDI/Sync
2. MIDI Drivers → Enable Nocturne
3. Create instrument track
4. Record arm the track
5. Play and record

### Generic DAW

1. Open preferences/settings
2. Find MIDI input/controller settings
3. Add device: "GestureSynth" or your MIDI port name
4. Assign to track and record

## Next Steps

- 🎹 Practice the exercises above
- 🔧 Read [CALIBRATION.md](CALIBRATION.md) to optimize for your setup
- 📊 Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand how it works
- 💻 See [DEVELOPMENT.md](DEVELOPMENT.md) if you want to extend features

## Getting Help

- 📖 Re-read this guide for techniques
- 🔍 Check `nocturne.log` for technical issues
- ⚙️ See [CALIBRATION.md](CALIBRATION.md) for tuning
- 💬 Open an issue on GitHub

---

**Have fun making music! 🎵**
