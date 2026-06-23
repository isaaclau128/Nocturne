import cv2
import sys
from core import VisionTracker, MusicProcessor, MIDIDriver
from core.config import get_config
from core.logging_config import setup_logging


# Setup logging first
logger = setup_logging('nocturne')


def draw_pitch_zones(frame, num_zones, config):
    """Draw horizontal lines indicating pitch zones."""
    try:
        h, w, _ = frame.shape
        zone_color = tuple(config.get('display.pitch_zone_color', [255, 255, 255]))
        
        for i in range(num_zones):
            y_line = int(h * (0.2 + (i * 0.085)))
            cv2.line(frame, (0, y_line), (w, y_line), zone_color, 1)
            cv2.putText(frame, f"Note {i}", (10, y_line - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1)
    except Exception as e:
        logger.warning(f"Error drawing pitch zones: {e}")


def draw_hand_position(frame, hand_landmarks, config):
    """Draw hand tracking visualization."""
    try:
        if not hand_landmarks:
            return
        
        h, w, _ = frame.shape
        hand_color = tuple(config.get('display.hand_position_color', [255, 0, 0]))
        
        index_tip = hand_landmarks.landmark[8]
        cx, cy = int(index_tip.x * w), int(index_tip.y * h)
        cv2.circle(frame, (cx, cy), 10, hand_color, -1)
        cv2.putText(frame, f"Hand Y: {round(index_tip.y, 2)}", 
                    (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, hand_color, 2)
    except Exception as e:
        logger.warning(f"Error drawing hand position: {e}")


def draw_dynamics(frame, velocity, config):
    """Draw velocity/dynamics feedback."""
    try:
        info_color = tuple(config.get('display.info_color', [0, 255, 0]))
        cv2.putText(frame, f"Dynamics (Velocity): {velocity}", 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, info_color, 2)
    except Exception as e:
        logger.warning(f"Error drawing dynamics: {e}")


def main():
    """Main application loop for gesture-based music synthesis."""
    
    vision = None
    processor = None
    midi = None
    cap = None
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = get_config()
        window_title = config.get('display.window_title', 'Nocturne - Gesture Synth')
        num_notes = config.get('music.num_notes', 7)
        
        # Initialize components
        logger.info("Initializing components...")
        vision = VisionTracker()
        processor = MusicProcessor()
        midi = MIDIDriver()
        
        # Setup webcam
        logger.info("Attempting to open webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot access webcam. Make sure a camera is connected.")
            return False
        
        logger.info("Webcam opened successfully")
        logger.info("Nocturne AI Instrument Running... Press 'q' to quit.")
        
        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                logger.warning("Failed to read frame from webcam")
                break
            
            frame_count += 1
            
            try:
                # Process frame through vision tracker
                frame, hand_results, face_results = vision.process_frame(frame)
                
                # Extract gestures and convert to MIDI
                control_data = processor.process(hand_results, face_results)
                note = control_data.get("pitch")
                velocity = control_data.get("velocity", 0)
                
                # Send MIDI
                if note is not None:
                    midi.send_note(note, velocity)
                
                # Draw visualization
                draw_pitch_zones(frame, num_notes, config)
                
                if hand_results and hand_results.multi_hand_landmarks:
                    draw_hand_position(frame, hand_results.multi_hand_landmarks[0], config)
                
                draw_dynamics(frame, velocity, config)
                
                # Display
                cv2.imshow(window_title, frame)
                
                # Log every 300 frames (~10 seconds at 30fps)
                if frame_count % 300 == 0:
                    logger.debug(f"Processed {frame_count} frames. Current: Note={note}, Velocity={velocity}")
                
                # Exit on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested shutdown (pressed 'q')")
                    break
                    
            except Exception as e:
                logger.error(f"Error in main loop (frame {frame_count}): {e}", exc_info=True)
                # Continue processing despite errors
                continue
        
        logger.info(f"Shutdown complete. Processed {frame_count} total frames.")
        return True
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
        return True
        
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}", exc_info=True)
        return False
        
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        try:
            if cap:
                cap.release()
                logger.debug("Webcam released")
        except Exception as e:
            logger.error(f"Error releasing webcam: {e}")
        
        try:
            if midi:
                midi.close()
        except Exception as e:
            logger.error(f"Error closing MIDI: {e}")
        
        try:
            if vision:
                vision.release()
        except Exception as e:
            logger.error(f"Error releasing vision tracker: {e}")
        
        try:
            cv2.destroyAllWindows()
            logger.debug("OpenCV windows closed")
        except Exception as e:
            logger.error(f"Error closing windows: {e}")
        
        logger.info("Cleanup complete")


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


