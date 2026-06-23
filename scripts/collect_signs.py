"""Simple data collection script to record hand landmarks with labels.

Run and press a key (a..z) to record a labeled sample. Press ESC to quit.
Saves JSON lines to `data/signs.jsonl` with {label: str, landmarks: [x,y,...]}
"""
import json
from pathlib import Path
import cv2
from core.vision import VisionTracker


OUT = Path('data')
OUT.mkdir(exist_ok=True)
OUT_FILE = OUT / 'signs.jsonl'


def lm_to_list(lm):
    return [lm.x for lm in lm] if hasattr(lm, '__iter__') and not hasattr(lm[0], 'x') else None


def main():
    vt = VisionTracker()
    print('Press a letter key to label the current hand pose, ESC to exit.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Unable to open webcam')
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('Failed to read frame')
                break
            frame, hand_results, face_results = vt.process_frame(frame)
            display = frame.copy()

            if hand_results and hand_results.multi_hand_landmarks:
                # draw a visual
                cv2.putText(display, 'Hand detected', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            cv2.imshow('collect_signs', display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if 97 <= key <= 122:  # a..z
                label = chr(key)
                if hand_results and hand_results.multi_hand_landmarks:
                    lm = hand_results.multi_hand_landmarks[0]
                    landmarks = []
                    for p in lm.landmark:
                        landmarks.extend([p.x, p.y])
                    with open(OUT_FILE, 'a') as f:
                        json.dump({'label': label, 'landmarks': landmarks}, f)
                        f.write('\n')
                    print(f'Saved sample for label: {label}')

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
