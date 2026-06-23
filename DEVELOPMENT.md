# Development Guide

Contribute to Nocturne and help make gesture-based music synthesis better.

## Table of Contents

- [Project Structure](#project-structure)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Making Changes](#making-changes)
- [Common Tasks](#common-tasks)
- [Debugging](#debugging)

## Project Structure

```
Nocturne/
├── main.py                    # Application entry point
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── .gitignore               # Git exclusions
├── LICENSE                  # MIT License
├── README.md                # Project overview
│
├── core/                    # Core modules
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management (ConfigManager)
│   ├── logging_config.py   # Logging setup (setup_logging, get_logger)
│   ├── vision.py           # Vision tracking (VisionTracker)
│   ├── processor.py        # Gesture processing (MusicProcessor)
│   └── midi_driver         # MIDI output (MIDIDriver)
│
├── docs/                   # Documentation
│   ├── SETUP.md           # Installation guide
│   ├── USAGE.md           # Playing guide
│   ├── CALIBRATION.md     # Configuration tuning
│   ├── ARCHITECTURE.md    # System design
│   └── DEVELOPMENT.md     # This file
│
└── tests/                 # Unit tests (planned)
    ├── test_config.py
    ├── test_processor.py
    └── test_vision.py
```

## Development Setup

### 1. Clone & Setup

```bash
git clone https://github.com/isaaclau128/Nocturne.git
cd Nocturne
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Install Development Tools

```bash
pip install pytest pytest-cov black pylint pytest-timeout
```

### 3. Pre-commit Hook (optional)

Automatically format code before commits:

```bash
pip install pre-commit
pre-commit install
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/pylint
    rev: 2.17.5
    hooks:
      - id: pylint
```

## Code Style

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these guidelines:

**Formatting:**
```bash
# Format all Python files
black core/ main.py

# Check formatting without changing
black --check core/ main.py
```

**Linting:**
```bash
# Check code quality
pylint core/
```

**Example:**
```python
# Good ✓
def get_hand_position(hand_landmarks, frame_height, frame_width):
    """Extract hand position from landmarks."""
    if not hand_landmarks:
        return None
    
    index_tip = hand_landmarks.landmark[8]
    x = int(index_tip.x * frame_width)
    y = int(index_tip.y * frame_height)
    return x, y


# Avoid ✗
def getHandPos(hand_lm, h, w):
    if hand_lm is not None:
        idx = hand_lm.landmark[8]
        return int(idx.x*w), int(idx.y*h)
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_pitch(hand_landmarks, notes):
    """
    Calculate MIDI pitch from hand landmarks.
    
    Maps hand height to one of the available notes. If landmarks
    are None, returns None.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object or None
        notes: List of MIDI note numbers [60, 62, 64, ...]
    
    Returns:
        int: MIDI note number (0-127) or None if no hand detected
        
    Raises:
        ValueError: If notes list is empty
        
    Example:
        >>> notes = [60, 62, 64, 65, 67, 69, 71]
        >>> pitch = calculate_pitch(landmarks, notes)
        >>> assert 60 <= pitch <= 71
    """
    if not hand_landmarks:
        return None
    # ... implementation
```

### Comments

- **Avoid obvious comments:** Code should be self-documenting
- **Explain why, not what:** "Why" comments are valuable
- **Update comments with code:** Outdated comments are worse than none

```python
# Avoid ✗
x = x + 1  # Add one to x

# Good ✓
# Offset hand x by camera distortion correction
x = x + self.calibration_offset
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_processor.py

# Run with coverage
pytest --cov=core tests/

# Run with verbose output
pytest -v

# Run with timeout (5 second per test)
pytest --timeout=5
```

### Writing Tests

Example test file: `tests/test_processor.py`

```python
import pytest
from core.processor import MusicProcessor
from core.config import ConfigManager


class TestMusicProcessor:
    """Test suite for MusicProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance for testing."""
        return MusicProcessor()
    
    def test_initialization(self, processor):
        """Test processor initializes with correct notes."""
        assert len(processor.solfeggio_notes) == 7
        assert processor.solfeggio_notes[0] == 60  # Do/C
    
    def test_dynamics_range(self, processor):
        """Test velocity calculations stay within bounds."""
        # Mock face landmarks
        class MockLandmark:
            def __init__(self, y_val):
                self.y = y_val
        
        class MockFace:
            landmark = {13: MockLandmark(0.5), 14: MockLandmark(0.45)}
        
        velocity = processor.get_dynamics(MockFace())
        assert 0 <= velocity <= 127
    
    def test_no_hand_returns_none(self, processor):
        """Test pitch returns None when hand not detected."""
        pitch = processor.get_pitch(None, None)
        assert pitch is None
```

## Making Changes

### Feature Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/my-feature-name
   ```

2. **Make changes:**
   - Follow code style guide
   - Add tests for new functionality
   - Update docstrings

3. **Test locally:**
   ```bash
   pytest
   black core/
   pylint core/
   ```

4. **Commit with meaningful message:**
   ```bash
   git add .
   git commit -m "feat: add multi-hand support to gesture processor"
   ```

5. **Push and create pull request:**
   ```bash
   git push origin feature/my-feature-name
   ```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation changes
- `style:` — Code style (formatting, semicolons, etc.)
- `refactor:` — Code restructuring without behavior change
- `perf:` — Performance improvements
- `test:` — Test additions/modifications
- `chore:` — Build, dependencies, tooling

**Examples:**
```
feat(processor): add chord recognition for multiple hands

fix(vision): handle edge case when face partially off-screen

docs: expand USAGE.md with advanced techniques

refactor(config): simplify ConfigManager initialization
```

## Common Tasks

### Adding a New Configuration Parameter

1. **Add to config.json:**
   ```json
   {
     "myfeature": {
       "enabled": true,
       "threshold": 0.5
     }
   }
   ```

2. **Access in code:**
   ```python
   from core.config import get_config
   
   config = get_config()
   is_enabled = config.get('myfeature.enabled')
   threshold = config.get('myfeature.threshold', default=0.5)
   ```

3. **Add documentation in CALIBRATION.md**

### Adding a New Logging Statement

```python
from core.logging_config import get_logger

logger = get_logger(__name__)

logger.debug("Detailed diagnostic info")
logger.info("Important event occurred")
logger.warning("Something unexpected")
logger.error("Error occurred", exc_info=True)
```

**Levels:**
- `DEBUG` — Verbose, frame-by-frame details
- `INFO` — Initialization, events, state changes
- `WARNING` — Recoverable issues, degraded functionality
- `ERROR` — Failures, exceptions (include `exc_info=True`)

### Adding Error Handling

```python
def process_frame(frame):
    """Process frame with error handling."""
    try:
        result = expensive_operation(frame)
        logger.debug(f"Processing complete: {result}")
        return result
    
    except ValueError as e:
        logger.warning(f"Invalid frame: {e}")
        return None  # Graceful fallback
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise  # Re-raise if truly fatal
```

## Debugging

### Enable Debug Logging

```json
{
  "logging": {
    "level": "DEBUG"
  }
}
```

View logs:
```bash
tail -f nocturne.log | grep "processor"  # Filter by module
```

### Using Python Debugger

```python
import pdb

def problematic_function():
    pdb.set_trace()  # Execution pauses here
    # Now you can inspect variables, step through code
```

Or use IDE debugger (VS Code, PyCharm, etc.)

### Print Debugging

```python
logger.debug(f"Hand landmarks: {hand_landmarks}")
logger.debug(f"Pitch calculated: {pitch}, Velocity: {velocity}")
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... code to profile ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

## Architecture Overview

```
Input: Webcam Frame
    ↓
VisionTracker
├─ MediaPipe Hands (hand landmarks)
├─ MediaPipe FaceMesh (face landmarks, iris)
└─ Output: detection results
    ↓
MusicProcessor
├─ get_pitch() — hand height → MIDI note
├─ get_dynamics() — mouth aperture → velocity
├─ get_accidental() — eye gaze → ±1 semitone
└─ Output: {pitch, velocity}
    ↓
MIDIDriver
├─ Filter by velocity threshold
├─ Generate Note On/Note Off messages
└─ Output: MIDI to synthesizer
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Performance Considerations

- **Frame processing:** Target 30 FPS
- **Gesture detection:** Sub-20ms latency for natural feel
- **Memory:** Keep frame buffer minimal
- **CPU:** Profile hot paths with cProfile

## Contributing Guidelines

1. **Follow code style** — Run `black` and `pylint`
2. **Add tests** — Maintain or improve coverage
3. **Update docs** — Reflect changes in markdown files
4. **Test manually** — Verify gesture recognition still works
5. **Meaningful commits** — Clear commit messages
6. **Keep scope focused** — One feature per PR

## Roadmap

Features planned for future releases:

- [ ] Multi-hand support (chords)
- [ ] Gesture recording/playback
- [ ] Custom gesture training
- [ ] Web-based UI
- [ ] Audio synthesis (built-in sounds)
- [ ] Mobile app support
- [ ] Machine learning gestures

## Getting Help

- 📖 Read [ARCHITECTURE.md](ARCHITECTURE.md) for design
- 🔍 Check existing code for examples
- 📊 View logs in `nocturne.log` for issues
- 💬 Open GitHub issues with questions

## Submitting Pull Requests

1. Fork repository
2. Create feature branch: `git checkout -b feature/name`
3. Make changes following guidelines
4. Commit with clear messages
5. Push to fork: `git push origin feature/name`
6. Open PR with description of changes
7. Address review feedback
8. Merge once approved!

---

**Thanks for contributing to Nocturne! 🎵**
