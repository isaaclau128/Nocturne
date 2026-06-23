# Setup & Installation Guide

Complete installation instructions for Nocturne on all supported platforms.

## System Requirements

- **Python:** 3.9, 3.10, 3.11, or 3.12
- **Webcam:** 720p or higher (USB or built-in)
- **RAM:** 4GB minimum, 8GB recommended
- **CPU:** Dual-core 2GHz or better
- **OS:** Linux, macOS, or Windows

## Prerequisites

### Windows

1. **Download Python:** Visit [python.org](https://www.python.org/downloads/) and download Python 3.11+
2. **Install:** Check "Add Python to PATH" during installation
3. **Verify:** Open Command Prompt and run:
   ```bash
   python --version
   ```

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python:**
   ```bash
   brew install python@3.11
   ```

3. **Verify:**
   ```bash
   python3 --version
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3-pip
python3 --version
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install python3.11
python3 --version
```

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/isaaclau128/Nocturne.git
cd Nocturne
```

Or if using SSH:
```bash
git clone git@github.com:isaaclau128/Nocturne.git
cd Nocturne
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `opencv-python` — Webcam and visualization
- `mediapipe` — Hand/face detection
- `python-rtmidi` — MIDI output
- `numpy` — Numerical calculations
- `scikit-learn` — Data processing

Installation should take 2-5 minutes depending on internet speed.

### 5. Verify Installation

```bash
python -c "import cv2, mediapipe, rtmidi, numpy; print('✓ All imports successful')"
```

## MIDI Setup

### Without External Synthesizer

If you don't have a hardware synthesizer, you can still use Nocturne with a software synthesizer:

**Windows:**
- Install [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) to create virtual MIDI ports
- Nocturne will automatically detect and use them

**macOS:**
- Virtual MIDI ports are built-in; no additional software needed

**Linux:**
- MIDI ports are created automatically

### With External Synthesizer

1. **Hardware Synth:** Connect via USB or MIDI adapter
2. **DAW (Ableton, FL Studio, etc.):** Set as MIDI input device
3. **Nocturne:** Automatically detects available MIDI ports and connects to the first one

To see available MIDI ports:
```bash
python -c "import rtmidi; m = rtmidi.MidiOut(); print(m.get_ports())"
```

## Running Nocturne

### Basic Launch

```bash
python main.py
```

The application will:
1. Load configuration from `config.json`
2. Initialize vision tracking (MediaPipe)
3. Open your webcam
4. Start listening for gestures
5. Send MIDI to connected synthesizer

### Exit

Press `q` in the video window or Ctrl+C in terminal.

### Logs

Application logs are written to `nocturne.log` with automatic rotation:
- Max file size: 5MB
- Keeps last 3 backup files
- Log level: INFO (set in `config.json`)

View logs in real-time:
```bash
# macOS/Linux:
tail -f nocturne.log

# Windows (PowerShell):
Get-Content -Path nocturne.log -Wait
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"

**Solution:** Reinstall dependencies
```bash
pip install --upgrade -r requirements.txt --force-reinstall
```

### "libGL.so.1 not found" (Linux)

**Solution:** Install OpenGL libraries
```bash
sudo apt-get install libgl1-mesa-glx
```

### Webcam Not Detected

1. Check webcam is connected: `ls /dev/video*` (Linux) or Device Manager (Windows)
2. Try different USB port
3. Close other camera applications (Zoom, Teams, etc.)
4. Restart application

### MIDI Port Not Found

1. Check MIDI device is connected/running
2. List available ports:
   ```bash
   python -c "import rtmidi; m = rtmidi.MidiOut(); print('Available ports:', m.get_ports())"
   ```
3. If empty, create virtual port:
   - **Windows:** Install loopMIDI
   - **macOS:** Built-in support
   - **Linux:** Use `amidi -l` to check ports

### Poor Hand/Face Detection

1. Improve lighting (good natural or artificial light)
2. Adjust thresholds in `config.json`
3. See [CALIBRATION.md](CALIBRATION.md) for detailed tuning

### Performance Issues

1. Reduce webcam resolution in camera settings
2. Lower logging level to "WARNING" in `config.json`
3. Check CPU usage: `top` (Linux/macOS) or Task Manager (Windows)

## Development Setup

For contributing or modifying Nocturne:

```bash
# Install with development dependencies
pip install -r requirements.txt
# Plus testing tools (if desired):
pip install pytest pytest-cov black pylint
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for details.

## Next Steps

1. Read [USAGE.md](USAGE.md) to learn how to play
2. Review [CALIBRATION.md](CALIBRATION.md) to tune sensitivity
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system

## Getting Help

- 📖 See [USAGE.md](USAGE.md) for common questions
- 🔧 See [CALIBRATION.md](CALIBRATION.md) for performance tuning
- 🐛 Check `nocturne.log` for error messages
- 💬 Open an issue on GitHub

---

**Happy playing! 🎵**
