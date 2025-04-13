import cv2
import numpy as np
import pyautogui
import time  # For managing cooldowns

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


class PinchDetector:
    def __init__(self, min_det_conf=0.7):
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe is not installed. Install it for pinch detection.")

        # Initialize MediaPipe hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Gesture thresholds (normalized distance values)
        self.pinch_threshold = 0.05          # For thumb-index pinch (drag)
        self.double_click_threshold = 0.05   # For thumb-middle double click
        self.scroll_threshold = 0.05         # For scroll gestures (thumb with ring or pinky)

    def detect_gestures(self, frame):
        """
        Processes an image frame to detect hand gestures.

        Returns:
          success: bool – True if a hand is detected.
          x_index: float – Normalized x-coordinate (0 to 1) of the index fingertip.
          y_index: float – Normalized y-coordinate (0 to 1) of the index fingertip.
          pinch_state: str – 'full_pinch' if thumb-index are together (for dragging) or 'half_pinch' if not.
          double_click_detected: bool – True when thumb and middle finger are very close.
          scroll_down_detected: bool – True when thumb and ring finger are close.
          scroll_up_detected: bool – True when thumb and pinky are close.
          open_hand_detected: bool – True when the hand is open (stop gesture).
          fist_detected: bool – True when a fist gesture is detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return False, 0, 0, 'no_hand', False, False, False, False, False

        # Process the first detected hand.
        hand_landmarks = results.multi_hand_landmarks[0]

        # Retrieve required landmarks:
        index_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        middle_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # --- Drag Gesture (Thumb and Index) ---
        pinch_dist = np.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 +
                             (index_finger_tip.y - thumb_tip.y) ** 2)
        pinch_state = 'full_pinch' if pinch_dist < self.pinch_threshold else 'half_pinch'

        # --- Double Click Gesture (Thumb and Middle) ---
        double_click_dist = np.sqrt((thumb_tip.x - middle_finger_tip.x) ** 2 +
                                    (thumb_tip.y - middle_finger_tip.y) ** 2)
        double_click_detected = double_click_dist < self.double_click_threshold

        # --- Scroll Gestures ---
        # Scroll down when thumb and ring finger are close
        scroll_down_dist = np.sqrt((thumb_tip.x - ring_finger_tip.x) ** 2 +
                                   (thumb_tip.y - ring_finger_tip.y) ** 2)
        scroll_down_detected = scroll_down_dist < self.scroll_threshold

        # Scroll up when thumb and pinky are close
        scroll_up_dist = np.sqrt((thumb_tip.x - pinky_tip.x) ** 2 +
                                 (thumb_tip.y - pinky_tip.y) ** 2)
        scroll_up_detected = scroll_up_dist < self.scroll_threshold

        # --- Open Hand (Stop) Gesture ---
        # For an open hand, check that the index, middle, ring and pinky fingertips are above their respective PIP joints
        index_open = index_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        middle_open = middle_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        ring_open = ring_finger_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y
        pinky_open = pinky_tip.y < hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y
        open_hand_detected = index_open and middle_open and ring_open and pinky_open

        # --- Fist Gesture ---
        # For a fist, check that the index, middle, ring, and pinky fingertips are below their respective PIP joints
        fist_detected = (index_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
                         middle_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                         ring_finger_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y and
                         pinky_tip.y > hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y)

        return True, index_finger_tip.x, index_finger_tip.y, pinch_state, double_click_detected, scroll_down_detected, scroll_up_detected, open_hand_detected, fist_detected


def pinch_control_demo():
    cap = cv2.VideoCapture(0)
    # Set frame dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    detector = PinchDetector()
    screen_w, screen_h = pyautogui.size()

    # States for dragging, double clicking, scrolling
    is_dragging = False
    double_click_active = False
    scroll_down_active = False
    scroll_up_active = False
    last_scroll_time = time.time()
    scroll_cooldown = 0.3  # Minimum time between scroll events (seconds)
    scroll_amount = 200    # Fixed scroll units (tweak as needed)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror-like effect.
        frame = cv2.flip(frame, 1)
        current_time = time.time()

        # Detect gestures in the frame.
        (success, x_index, y_index, pinch_state, double_click_detected,
         scroll_down_detected, scroll_up_detected, open_hand_detected,
         fist_detected) = detector.detect_gestures(frame)
        # Convert normalized index fingertip coordinates to screen pixel coordinates.
        mouse_x = int(x_index * screen_w)
        mouse_y = int(y_index * screen_h)

        # --- Mouse Movement and Dragging ---
        if success:
            if pinch_state == 'half_pinch':
                pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)
                if is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False
            elif pinch_state == 'full_pinch':
                if not is_dragging:
                    pyautogui.mouseDown()
                    is_dragging = True
                pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)
        else:
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False

        # --- Double Click Handling ---
        if double_click_detected and not double_click_active:
            pyautogui.doubleClick()
            double_click_active = True
        elif not double_click_detected:
            double_click_active = False

        # --- Scroll Gesture Handling ---
        if success and pinch_state == 'half_pinch':
            if scroll_down_detected and not scroll_down_active and (current_time - last_scroll_time) > scroll_cooldown:
                pyautogui.scroll(-scroll_amount, x=mouse_x, y=mouse_y)
                scroll_down_active = True
                last_scroll_time = current_time
            elif not scroll_down_detected:
                scroll_down_active = False

            if scroll_up_detected and not scroll_up_active and (current_time - last_scroll_time) > scroll_cooldown:
                pyautogui.scroll(scroll_amount, x=mouse_x, y=mouse_y)
                scroll_up_active = True
                last_scroll_time = current_time
            elif not scroll_up_detected:
                scroll_up_active = False

        # --- Visual Feedback ---
        # Always show the live feed.
        display_frame = frame

        gesture_text = f"Pinch: {pinch_state}"
        if double_click_detected:
            gesture_text += " | Double Click"
        if scroll_down_detected:
            gesture_text += " | Scroll Down"
        if scroll_up_detected:
            gesture_text += " | Scroll Up"
        if open_hand_detected:
            gesture_text += " | STOP"
        cv2.putText(display_frame, gesture_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Pinch Control Demo", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pinch_control_demo()