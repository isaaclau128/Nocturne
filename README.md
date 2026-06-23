# Nocturne 🎵

**Real-time gesture-based music synthesis using computer vision and AI.**

Nocturne transforms your hand, eye, and mouth movements into expressive MIDI music. Control pitch with hand height, dynamics with mouth opening, and articulation with eye gaze—all in real-time.

## Features

- **Hand-based pitch control** — Position your hand vertically to select from 7 solfeggio notes (Do, Re, Mi, Fa, Sol, La, Ti)
- **Mouth-based dynamics** — Open your mouth to control note velocity (0-127)
- **Eye-based articulation** — Gaze left for flats, right for sharps
- **MIDI output** — Connect to any synthesizer, DAW, or MIDI device
- **Real-time processing** — Sub-20ms latency for natural musical expression
- **Fully configurable** — Adjust all thresholds and sensitivity via `config.json`
- **Comprehensive logging** — Debug issues with detailed logs and frame counters

## Quick Start

### Installation

See [SETUP.md](SETUP.md) for detailed OS-specific installation instructions.

```bash
# Clone repository
git clone https://github.com/isaaclau128/Nocturne.git
cd Nocturne

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
python main.py
```

Press `q` to quit. Logs are written to `nocturne.log`.

## How to Play

See [USAGE.md](USAGE.md) for comprehensive playing guide.

### Quick controls:

| Body Part | Effect |
|-----------|--------|
| **Hand Height** | Select note (7 zones, low=Do, high=Ti) |
| **Mouth Open** | Control volume (velocity 0-127) |
| **Eyes Left** | Add flat (-1 semitone) |
| **Eyes Right** | Add sharp (+1 semitone) |

## Architecture

```
main.py
├── VisionTracker (core/vision.py)
│   ├── Hand detection (MediaPipe Hands)
│   └── Face detection (MediaPipe FaceMesh)
├── MusicProcessor (core/processor.py)
│   ├── Gesture → MIDI mapping
│   ├── Pitch calculation (hand height)
│   ├── Velocity calculation (mouth opening)
│   └── Accidental calculation (eye gaze)
└── MIDIDriver (core/midi_driver)
    └── MIDI Note On/Off messages
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Configuration

Edit `config.json` to customize behavior:

```json
{
  "music": {
    "solfeggio_notes": [60, 62, 64, 65, 67, 69, 71],
    "mouth_aperture_range": [0.01, 0.08],
    "hand_height_range": [0.2, 0.8],
    "eye_gaze_left_threshold": 0.35,
    "eye_gaze_right_threshold": 0.65
  },
  "logging": {
    "level": "INFO",
    "log_file": "nocturne.log"
  }
}
```

See [CALIBRATION.md](CALIBRATION.md) for tuning guide.

## System Requirements

- **Python** 3.9+
- **Webcam** (720p or higher recommended)
- **MIDI synthesizer or DAW** (optional, for audio output)
- **Linux/macOS/Windows**

## Project Structure

```
Nocturne/
├── main.py                      # Application entry point
├── config.json                  # Configuration file
├── requirements.txt             # Python dependencies
├── core/
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration management
│   ├── logging_config.py       # Logging setup
│   ├── vision.py               # Vision tracking (MediaPipe)
│   ├── processor.py            # Gesture processing
│   └── midi_driver             # MIDI output
├── nocturne.log                # Application logs (auto-generated)
└── docs/
    ├── SETUP.md                # Installation guide
    ├── USAGE.md                # Playing guide
    ├── CALIBRATION.md          # Configuration tuning
    ├── ARCHITECTURE.md         # System design
    └── DEVELOPMENT.md          # Contributing guide
```

## Troubleshooting

- **No hand/face detected?** → Ensure good lighting, adjust detection thresholds in config.json
- **MIDI not working?** → Check MIDI port availability, see SETUP.md for your OS
- **Poor gesture recognition?** → See CALIBRATION.md for sensitivity tuning
- **Performance issues?** → Reduce frame resolution in camera settings

See [USAGE.md](USAGE.md) for more troubleshooting tips.

## Dependencies

| Library | Purpose |
|---------|---------|
| **OpenCV** | Webcam frame capture and visualization |
| **MediaPipe** | Hand and face landmark detection |
| **python-rtmidi** | MIDI output to synthesizers |
| **NumPy** | Numerical computations |
| **scikit-learn** | Potential future ML enhancements |

## Contributing

See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use Nocturne in research or projects, please cite:

```bibtex
@software{nocturne2024,
  title={Nocturne: Gesture-Based Music Synthesis},
  author={Isaac Lau},
  year={2024},
  url={https://github.com/isaaclau128/Nocturne}
}
```

## Roadmap

- [ ] Multiple hand support
- [ ] Chord generation
- [ ] Gesture recording/playback
- [ ] Web-based UI
- [ ] Mobile app (iOS/Android)
- [ ] ML-based gesture learning
- [ ] Audio synthesis (eliminate external synth dependency)

## Support

- 📖 Check [USAGE.md](USAGE.md) for common questions
- 🔧 See [CALIBRATION.md](CALIBRATION.md) for performance tuning
- 🛠️ Review logs in `nocturne.log` for debugging
- 💬 Open an issue on GitHub

---

**Made with 🎵 and computer vision**
