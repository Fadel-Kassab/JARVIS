import streamlit as st
import numpy as np
import pyautogui
import time
import cv2
import threading
import platform # Used for platform-specific checks if needed later, but not for xdotool
import plotly.graph_objects as go
# from streamlit_plotly import plotly_chart # Not needed, st.plotly_chart works
from PIL import Image, ImageDraw, ImageFont
import math # For hypot function

# Attempt to import MediaPipe
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    st.error("MediaPipe library not found. Please install it: pip install mediapipe")
    MP_AVAILABLE = False
    # Optionally exit if MediaPipe is essential
    # st.stop()

# --- Configuration ---
# Disable pyautogui's fail-safe - Use with caution! Allows control near screen edges.
# Move the mouse quickly to a corner of the screen to trigger the fail-safe if needed.
pyautogui.FAILSAFE = True # Keep failsafe enabled by default for safety
# You can set it to False if you understand the risks: pyautogui.FAILSAFE = False

# --- Pinch Detector Class ---
class PinchDetector:
    def __init__(self, min_det_conf=0.7, min_track_conf=0.6):
        if not MP_AVAILABLE:
            raise ImportError("MediaPipe is not installed or failed to import.")

        self.mp_hands = mp.solutions.hands
        # Initialize MediaPipe Hands with adjusted confidence levels
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,      # Process video stream
            max_num_hands=1,              # Detect only one hand
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf # Lower tracking confidence can sometimes help keep track
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Gesture thresholds (normalized distances) - Tuned values
        self.pinch_threshold = 0.05        # Distance for thumb-index pinch (drag)
        self.double_click_threshold = 0.04 # Distance for thumb-middle pinch (double click)
        self.scroll_threshold = 0.045      # Distance for thumb-ring/pinky pinch (scroll)
        self.thumb_wrist_threshold = 0.15  # Minimum thumb distance from wrist for scroll confidence

        # Smoothing parameters
        self.smoothing_factor = 0.6        # Lower value = more smoothing, higher value = more responsive
        self.prev_x, self.prev_y = 0.5, 0.5 # Initialize previous position to center

        # Gesture state stabilization
        self.pinch_state_history = ['no_hand'] * 3 # History to stabilize pinch detection
        self.gesture_detected_history = {'double_click': [False]*3, 'scroll_down': [False]*3, 'scroll_up': [False]*3}


    def _calculate_distance(self, p1, p2):
        """Calculates Euclidean distance between two MediaPipe landmarks."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _update_gesture_history(self, gesture_name, detected):
        """Updates the history for a specific gesture."""
        self.gesture_detected_history[gesture_name].pop(0)
        self.gesture_detected_history[gesture_name].append(detected)
        # Require detection in multiple recent frames for stability
        return self.gesture_detected_history[gesture_name].count(True) >= 2


    def detect_gestures(self, frame):
        """
        Processes an image frame to detect hand landmarks and gestures.

        Returns:
         tuple: (
             frame_rgb: Processed frame with landmarks (NumPy array RGB),
             success: bool – True if a hand is detected.
             smooth_x: float – Smoothed normalized x-coordinate (0 to 1) of the index fingertip.
             smooth_y: float – Smoothed normalized y-coordinate (0 to 1) of the index fingertip.
             pinch_state: str – 'full_pinch', 'half_pinch', or 'no_hand'.
             double_click_stable: bool – True if double click gesture is stable.
             scroll_down_stable: bool – True if scroll down gesture is stable.
             scroll_up_stable: bool – True if scroll up gesture is stable.
         )
        """
        # Flip the frame horizontally for a later selfie-view display
        # And convert the BGR image to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True # Make it writeable again for drawing

        success = False
        x_index, y_index = self.prev_x, self.prev_y # Default to previous position if no hand
        pinch_state = 'no_hand'
        double_click_detected = False
        scroll_down_detected = False
        scroll_up_detected = False

        if results.multi_hand_landmarks:
            success = True
            # Process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks on the frame_rgb
            self.mp_draw.draw_landmarks(
                frame_rgb,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # Joints
                self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)  # Connections
            )

            # --- Get Key Landmarks ---
            try:
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            except IndexError:
                # Should not happen if results.multi_hand_landmarks is not empty, but good practice
                return frame_rgb, False, self.prev_x, self.prev_y, 'no_hand', False, False, False

            # --- Cursor Position Smoothing ---
            x_index = index_tip.x
            y_index = index_tip.y
            smooth_x = self.prev_x * self.smoothing_factor + x_index * (1 - self.smoothing_factor)
            smooth_y = self.prev_y * self.smoothing_factor + y_index * (1 - self.smoothing_factor)
            self.prev_x, self.prev_y = smooth_x, smooth_y # Update previous position

            # --- Pinch (Drag) Gesture (Thumb <-> Index) ---
            pinch_dist = self._calculate_distance(index_tip, thumb_tip)
            current_pinch_state = 'full_pinch' if pinch_dist < self.pinch_threshold else 'half_pinch'

            # Update pinch state history
            self.pinch_state_history.pop(0)
            self.pinch_state_history.append(current_pinch_state)

            # Determine stable pinch state based on history
            if self.pinch_state_history.count('full_pinch') >= 2:
                pinch_state = 'full_pinch'
            elif self.pinch_state_history.count('half_pinch') >= 2:
                pinch_state = 'half_pinch'
            else:
                pinch_state = self.pinch_state_history[1] # Keep previous stable state

            # --- Double Click Gesture (Thumb <-> Middle) ---
            double_click_dist = self._calculate_distance(thumb_tip, middle_tip)
            double_click_detected = double_click_dist < self.double_click_threshold

            # --- Scroll Gestures (Check thumb distance from wrist for confidence) ---
            thumb_wrist_dist = self._calculate_distance(thumb_tip, wrist)
            if thumb_wrist_dist > self.thumb_wrist_threshold:
                # Scroll Down (Thumb <-> Ring)
                scroll_down_dist = self._calculate_distance(thumb_tip, ring_tip)
                scroll_down_detected = scroll_down_dist < self.scroll_threshold

                # Scroll Up (Thumb <-> Pinky)
                scroll_up_dist = self._calculate_distance(thumb_tip, pinky_tip)
                scroll_up_detected = scroll_up_dist < self.scroll_threshold
            else:
                scroll_down_detected = False
                scroll_up_detected = False

        else: # No hand detected
             # Update pinch state history for no hand
            self.pinch_state_history.pop(0)
            self.pinch_state_history.append('no_hand')
            pinch_state = 'no_hand' if self.pinch_state_history.count('no_hand') >= 2 else self.pinch_state_history[1]


        # Stabilize gesture detection using history
        double_click_stable = self._update_gesture_history('double_click', double_click_detected)
        scroll_down_stable = self._update_gesture_history('scroll_down', scroll_down_detected)
        scroll_up_stable = self._update_gesture_history('scroll_up', scroll_up_detected)


        return frame_rgb, success, smooth_x, smooth_y, pinch_state, double_click_stable, scroll_down_stable, scroll_up_stable

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


# --- 3D Box Plotly Function ---
def create_3d_box(rotate_x=0, rotate_y=0, rotate_z=0):
    # Define vertices of a unit cube centered at origin
    vertices = np.array([
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5], # Bottom face (z=-0.5)
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]  # Top face (z=0.5)
    ])

    # Rotation matrices (degrees to radians)
    theta_x, theta_y, theta_z = np.radians(rotate_x), np.radians(rotate_y), np.radians(rotate_z)
    Rx = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])

    # Combined rotation (apply Z, then Y, then X)
    R = Rz @ Ry @ Rx
    rotated_vertices = vertices @ R.T # Apply rotation

    # Define faces using vertex indices (adjust for centered cube)
    faces = [
        [0, 1, 2, 3], # Bottom
        [4, 5, 6, 7], # Top
        [0, 1, 5, 4], # Front
        [2, 3, 7, 6], # Back
        [1, 2, 6, 5], # Right
        [0, 3, 7, 4]  # Left
    ]

    # Colors for faces
    colors = ['rgba(255, 0, 0, 0.6)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 0, 255, 0.6)',
              'rgba(255, 255, 0, 0.6)', 'rgba(255, 0, 255, 0.6)', 'rgba(0, 255, 255, 0.6)']

    fig = go.Figure()

    # Add faces
    for i, face_indices in enumerate(faces):
         fig.add_trace(go.Mesh3d(
            x=rotated_vertices[face_indices, 0],
            y=rotated_vertices[face_indices, 1],
            z=rotated_vertices[face_indices, 2],
            i=[0, 0, 1, 2], # Define triangles within the face
            j=[1, 2, 3, 3],
            k=[2, 3, 0, 0],
            color=colors[i % len(colors)],
            opacity=0.7,
            hoverinfo='none'
        ))

    # Add edges (optional, makes it look more wireframe)
    edges = [
        [0,1],[1,2],[2,3],[3,0], # bottom
        [4,5],[5,6],[6,7],[7,4], # top
        [0,4],[1,5],[2,6],[3,7]  # sides
    ]
    for edge in edges:
         fig.add_trace(go.Scatter3d(
            x=rotated_vertices[edge, 0],
            y=rotated_vertices[edge, 1],
            z=rotated_vertices[edge, 2],
            mode='lines', line=dict(color='black', width=3), hoverinfo='none'
        ))


    fig.update_layout(
        # title="3D Box - Drag with Gestures", title_font=dict(size=16, color='white'), # Title overlaps weirdly in streamlit sometimes
        margin=dict(l=10, r=10, b=10, t=10), # Reduced margin
        scene=dict(
            xaxis=dict(range=[-1, 1], visible=False), # Hide axes for cleaner look
            yaxis=dict(range=[-1, 1], visible=False),
            zaxis=dict(range=[-1, 1], visible=False),
            aspectmode='cube', # Ensure cube aspect ratio
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)) # Camera position
        ),
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Styled Message Function ---
def create_styled_message(text, icon="✨", color="#3498db", background="#f1f8fe"):
    width, height = 600, 80 # Adjusted height
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0)) # Transparent background
    draw = ImageDraw.Draw(image)

    try:
        # Try loading a common sans-serif font, adjust path if necessary
        # On Linux/macOS, common paths might be different. Provide a common name.
        font_path = "arial.ttf" # Common on Windows
        font = ImageFont.truetype(font_path, 20)
        small_font = ImageFont.truetype(font_path, 12)
    except IOError:
        # Fallback to default bitmap font if Arial isn't found
        print("Arial font not found, using default font. Install Arial for better visuals.")
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Parse colors (handle potential errors)
    try:
        r, g, b = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        r, g, b = 52, 152, 219 # Default blue
    try:
        bg_r, bg_g, bg_b = tuple(int(background.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
         bg_r, bg_g, bg_b = 241, 248, 254 # Default light blue


    # Draw rounded rectangle background
    draw.rounded_rectangle([(10, 10), (width - 10, height - 10)], radius=15, fill=(bg_r, bg_g, bg_b, 220)) # Slightly more opaque

    # Draw icon and text (vertically centered)
    icon_width = draw.textlength(icon, font=font) # Use textlength
    text_width = draw.textlength(text, font=font) # Use textlength

    icon_x, text_x = 30, 40 + icon_width
    total_content_width = (text_x - icon_x) + text_width
    start_x = (width - total_content_width) / 2 # Center horizontally

    # Adjust y position for vertical centering (approximate)
    y_pos = (height - font.getbbox("Aj")[3]) / 2 - 5 # Rough vertical center

    draw.text((start_x, y_pos), icon, fill=(r, g, b, 255), font=font)
    draw.text((start_x + icon_width + 10, y_pos), text, fill=(r, g, b, 255), font=font)

    # Draw small text at the bottom right
    app_text = "Gesture Control App"
    app_text_width = draw.textlength(app_text, font=small_font) # Use textlength
    draw.text((width - app_text_width - 20, height - 25), app_text, fill=(100, 100, 100, 200), font=small_font)

    return image


# --- Mouse Control Thread ---
def mouse_control_thread(stop_event, gesture_data_ref, screen_dims, rotation_ref):
    """
    Handles mouse control actions based on gesture data in a separate thread.
    Uses pyautogui for cross-platform compatibility.

    Args:
        stop_event (threading.Event): Event to signal thread termination.
        gesture_data_ref (dict): Mutable dictionary holding the latest gesture data from the main thread.
        screen_dims (tuple): Screen width and height (screen_w, screen_h).
        rotation_ref (dict): Mutable dictionary holding rotation angles {'x', 'y', 'z'}.
    """
    screen_w, screen_h = screen_dims
    is_dragging = False
    drag_start_screen_pos = None # Store initial screen position for drag calculations
    drag_start_norm_pos = None   # Store initial normalized position for rotation calculations

    # Action state tracking and cooldowns
    action_states = {'double_click': False, 'scroll_down': False, 'scroll_up': False}
    last_action_time = {'double_click': 0, 'scroll_down': 0, 'scroll_up': 0}
    action_cooldown = 0.3  # Cooldown in seconds between distinct actions (clicks, scrolls)
    scroll_repeat_cooldown = 0.1 # Cooldown for repeated scrolls while gesture is held
    scroll_amount = 40       # Amount to scroll per event (adjust for sensitivity)
    position_threshold = 3   # Minimum pixel movement to trigger mouse move update

    last_mouse_position = pyautogui.position() # Start with current mouse pos

    print("Mouse control thread started.")

    while not stop_event.is_set():
        try:
            # Safely get data from the shared dictionary
            # Make a local copy to avoid race conditions during processing
            local_gesture_data = gesture_data_ref.copy()

            success = local_gesture_data.get('success', False)
            norm_x = local_gesture_data.get('x_index', 0.5)
            norm_y = local_gesture_data.get('y_index', 0.5)
            pinch_state = local_gesture_data.get('pinch_state', 'no_hand')
            double_click_stable = local_gesture_data.get('double_click', False)
            scroll_down_stable = local_gesture_data.get('scroll_down', False)
            scroll_up_stable = local_gesture_data.get('scroll_up', False)

            current_time = time.time()

            # --- Calculate Target Screen Position ---
            # Invert Y for natural mapping (optional, depends on preference)
            # target_x = int(norm_x * screen_w)
            # target_y = int(norm_y * screen_h)
            # Flip X coordinate if camera is mirrored:
            target_x = int((1 - norm_x) * screen_w) # Flipped X
            target_y = int(norm_y * screen_h)       # Normal Y

            # Clamp coordinates to screen bounds
            target_x = max(0, min(screen_w - 1, target_x))
            target_y = max(0, min(screen_h - 1, target_y))

            # --- Mouse Movement ---
            # Only move if the hand is detected and position changed significantly
            pos_changed = math.hypot(target_x - last_mouse_position[0], target_y - last_mouse_position[1]) > position_threshold

            if success and pos_changed and not is_dragging: # Don't move cursor if dragging (drag handles it)
                pyautogui.moveTo(target_x, target_y, duration=0.01) # Small duration for smoothness
                last_mouse_position = (target_x, target_y)

            # --- Drag Handling (Full Pinch) ---
            if pinch_state == 'full_pinch':
                if not is_dragging:
                    # Start dragging
                    is_dragging = True
                    drag_start_screen_pos = (target_x, target_y)
                    drag_start_norm_pos = (norm_x, norm_y) # Store normalized pos for rotation
                    pyautogui.mouseDown(button='left')
                    print("Drag started")
                    # Move to the initial drag point precisely
                    pyautogui.moveTo(target_x, target_y, duration=0.01)
                    last_mouse_position = (target_x, target_y)
                else:
                    # Continue dragging - move mouse
                    if pos_changed:
                         pyautogui.moveTo(target_x, target_y, duration=0.01)
                         last_mouse_position = (target_x, target_y)

                         # --- Update 3D Box Rotation Based on Drag ---
                         if drag_start_norm_pos:
                             delta_norm_x = norm_x - drag_start_norm_pos[0]
                             delta_norm_y = norm_y - drag_start_norm_pos[1]
                             # Map delta to rotation (adjust sensitivity factor)
                             sensitivity = 360 # degrees per full screen drag
                             # Update shared rotation dictionary (thread-safe update not strictly needed for simple types)
                             rotation_ref['y'] += delta_norm_x * sensitivity # Horizontal drag -> Y rotation
                             rotation_ref['x'] -= delta_norm_y * sensitivity # Vertical drag -> X rotation (inverted y)
                             # Clamp rotation angles if desired (e.g., -180 to 180)
                             # rotation_ref['x'] = max(-180, min(180, rotation_ref['x']))
                             # rotation_ref['y'] = max(-180, min(180, rotation_ref['y']))
                             # Update start position for continuous rotation calculation
                             drag_start_norm_pos = (norm_x, norm_y)


            elif is_dragging:
                 # End dragging (pinch released or hand lost)
                pyautogui.mouseUp(button='left')
                is_dragging = False
                drag_start_screen_pos = None
                drag_start_norm_pos = None
                last_action_time['double_click'] = current_time # Add cooldown after drag ends
                print("Drag ended")


            # --- Other Actions (Only if not dragging) ---
            if not is_dragging:
                # Double Click
                if double_click_stable and not action_states['double_click'] and (current_time - last_action_time['double_click']) > action_cooldown:
                    pyautogui.doubleClick()
                    print("Double Click")
                    action_states['double_click'] = True
                    last_action_time['double_click'] = current_time
                    # Prevent other actions immediately after
                    last_action_time['scroll_down'] = current_time
                    last_action_time['scroll_up'] = current_time
                elif not double_click_stable:
                    action_states['double_click'] = False # Reset state when gesture stops


                # Scroll Down
                # Allow repeated scrolling if gesture is held
                scroll_cooldown = scroll_repeat_cooldown if action_states['scroll_down'] else action_cooldown
                if scroll_down_stable and (current_time - last_action_time['scroll_down']) > scroll_cooldown:
                    pyautogui.scroll(-scroll_amount) # Negative value for scroll down
                    print("Scroll Down")
                    action_states['scroll_down'] = True
                    last_action_time['scroll_down'] = current_time
                    # Prevent other actions immediately after
                    last_action_time['double_click'] = current_time
                    last_action_time['scroll_up'] = current_time
                elif not scroll_down_stable:
                     action_states['scroll_down'] = False # Reset state

                # Scroll Up
                scroll_cooldown = scroll_repeat_cooldown if action_states['scroll_up'] else action_cooldown
                if scroll_up_stable and (current_time - last_action_time['scroll_up']) > scroll_cooldown:
                    pyautogui.scroll(scroll_amount) # Positive value for scroll up
                    print("Scroll Up")
                    action_states['scroll_up'] = True
                    last_action_time['scroll_up'] = current_time
                    # Prevent other actions immediately after
                    last_action_time['double_click'] = current_time
                    last_action_time['scroll_down'] = current_time
                elif not scroll_up_stable:
                    action_states['scroll_up'] = False # Reset state


        except Exception as e:
            print(f"Error in mouse control thread: {e}")
            # Consider whether to break or continue based on the error
            # stop_event.set() # Example: stop on any error

        # Short sleep to prevent high CPU usage and allow other threads/processes
        time.sleep(0.01)

    print("Mouse control thread stopped.")


# --- Streamlit App ---
def pinch_control_app():
    st.set_page_config(
        page_title="Gesture Control", page_icon="👋", layout="wide", initial_sidebar_state="collapsed"
    )

    # Custom CSS (optional)
    st.markdown("""
    <style>
        /* Add custom styles here if needed */
        .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .stButton button { padding: 0.5rem 1rem; border-radius: 0.5rem; }
        .stImage > img { border: 2px solid #4CAF50; border-radius: 10px; } /* Style webcam feed */
    </style>
    """, unsafe_allow_html=True)

    st.title("👋 Gesture Control App")

    # Check MediaPipe availability again (in case it failed during initial import)
    if not MP_AVAILABLE:
         st.error("MediaPipe is required but not installed or failed to load. Please install it (`pip install mediapipe`) and restart.")
         st.stop() # Stop the Streamlit app if MediaPipe isn't available


    # App description
    st.markdown("""
    Control your mouse using hand gestures detected via your webcam.
    - **Index Finger:** Move Cursor
    - 👍 + 👉 (Thumb-Index Pinch): **Drag** (Hold pinch and move hand)
    - 👍 +  középső (Thumb-Middle Pinch): **Double Click**
    - 👍 + 💍 (Thumb-Ring Pinch): **Scroll Down**
    - 👍 + 🤙 (Thumb-Pinky Pinch): **Scroll Up**

    **Instructions:**
    1. Click **Start Control**.
    2. Allow webcam access if prompted.
    3. Minimize this browser window or switch to another application to control it.
    4. To stop, return to this window and click **Stop Control**.
    5. **Safety:** Quickly move the mouse to a screen corner to disable control if needed (PyAutoGUI Fail-Safe).
    """)

    # Initialize session state variables
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = PinchDetector()
        except ImportError as e:
            st.error(f"Failed to initialize PinchDetector: {e}")
            st.stop()
    if 'stop_event' not in st.session_state:
        st.session_state.stop_event = threading.Event()
    # Use a thread-safe way to share data if complex updates were needed,
    # but for simple dict reads/writes, direct access is often okay here.
    if 'gesture_data' not in st.session_state:
        st.session_state.gesture_data = {
            'success': False, 'x_index': 0.5, 'y_index': 0.5,
            'pinch_state': 'no_hand', 'double_click': False,
            'scroll_down': False, 'scroll_up': False
        }
    if 'control_thread' not in st.session_state:
        st.session_state.control_thread = None
    if 'camera' not in st.session_state:
         st.session_state.camera = None # To store the VideoCapture object

    # Shared dictionary for rotation (accessed by mouse thread)
    if 'rotation' not in st.session_state:
        st.session_state.rotation = {'x': 0, 'y': 0, 'z': 0}


    # --- UI Layout ---
    col_controls, col_demo = st.columns([1, 1])
    with col_controls:
        start_button_label = "Stop Control" if st.session_state.is_running else "Start Control"
        if st.button(start_button_label, key='start_stop_button'):
            if st.session_state.is_running:
                # --- Stop Sequence ---
                st.session_state.is_running = False
                st.session_state.stop_event.set() # Signal thread to stop
                if st.session_state.control_thread and st.session_state.control_thread.is_alive():
                    st.session_state.control_thread.join(timeout=1.0) # Wait for thread
                if st.session_state.camera and st.session_state.camera.isOpened():
                    st.session_state.camera.release() # Release camera
                    st.session_state.camera = None
                st.session_state.detector.close() # Clean up detector resources
                print("Control stopped.")
                st.rerun() # Force rerun to update button label etc.
            else:
                # --- Start Sequence ---
                st.session_state.camera = cv2.VideoCapture(0) # Try default camera
                if not st.session_state.camera.isOpened():
                     st.error("Could not open webcam. Please check if it's connected and not used by another app.")
                     st.session_state.camera = None
                else:
                    st.session_state.is_running = True
                    st.session_state.stop_event.clear()
                    st.session_state.rotation = {'x': 0, 'y': 0, 'z': 0} # Reset rotation on start

                    # Get screen dimensions once
                    try:
                        screen_w, screen_h = pyautogui.size()
                    except Exception as e:
                         st.error(f"Could not get screen size. PyAutoGUI might need configuration: {e}")
                         st.session_state.is_running = False
                         st.session_state.camera.release()
                         st.session_state.camera = None

                    if st.session_state.is_running: # Check again in case screen size failed
                        # (Re)Initialize detector if needed (e.g., after stopping)
                        if 'detector' not in st.session_state or st.session_state.detector is None:
                             st.session_state.detector = PinchDetector()

                        # Start the mouse control thread
                        st.session_state.control_thread = threading.Thread(
                            target=mouse_control_thread,
                            args=(st.session_state.stop_event,
                                  st.session_state.gesture_data, # Pass the dict directly
                                  (screen_w, screen_h),
                                  st.session_state.rotation # Pass the rotation dict
                                  ),
                            daemon=True # Allows main program to exit even if thread is running
                        )
                        st.session_state.control_thread.start()
                        print("Control started.")
                        st.rerun() # Rerun to start the video loop


    with col_demo:
        if st.button("Reset 3D Box", key="reset_box"):
            st.session_state.rotation = {'x': 0, 'y': 0, 'z': 0}
            # Force a rerun to update the chart immediately if needed
            # st.rerun() # Usually not necessary as Plotly chart updates automatically


    st.markdown("---") # Separator

    # --- Main Display Area ---
    col_feed, col_status, col_3d = st.columns([2, 1, 2]) # Adjust column ratios as needed

    with col_feed:
        st.subheader("📷 Webcam Feed")
        frame_placeholder = st.empty()
        if not st.session_state.is_running:
            frame_placeholder.info("Webcam feed will appear here when control is started.")
        elif st.session_state.camera is None or not st.session_state.camera.isOpened():
             frame_placeholder.error("Webcam not available.")


    with col_status:
        st.subheader("📊 Status")
        status_placeholder = st.empty()
        # Initial status message
        status_placeholder.markdown(
             f"""
             **Hand Detected:** No
             **Pinch State:** -
             **Double Click:** Off
             **Scroll Down:** Off
             **Scroll Up:** Off
             **Mouse Pos:** ({st.session_state.gesture_data['x_index']:.2f}, {st.session_state.gesture_data['y_index']:.2f})
             """
         )

    with col_3d:
         st.subheader("📦 3D Box Demo")
         st.markdown("_Pinch (👍+👉) and drag to rotate._")
         box_placeholder = st.empty()
         # Initial drawing of the box
         box_placeholder.plotly_chart(
            create_3d_box(st.session_state.rotation['x'], st.session_state.rotation['y'], st.session_state.rotation['z']),
            use_container_width=True
         )


    # --- Main Loop (Runs only when 'is_running' is True) ---
    if st.session_state.is_running and st.session_state.camera:
        while True:
            if st.session_state.stop_event.is_set(): # Check if stop was requested
                print("Stop event detected in main loop.")
                # Release camera if not already done (belt and suspenders)
                if st.session_state.camera and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
                    st.session_state.camera = None
                break # Exit the loop

            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Failed to grab frame from webcam. Stopping.")
                st.session_state.stop_event.set() # Signal thread to stop
                st.session_state.is_running = False
                if st.session_state.camera and st.session_state.camera.isOpened():
                    st.session_state.camera.release()
                st.rerun()
                break

            # --- Gesture Detection ---
            try:
                processed_frame_rgb, success, norm_x, norm_y, pinch_state, dbl_click, scr_down, scr_up = st.session_state.detector.detect_gestures(frame)
            except Exception as e:
                 st.error(f"Error during gesture detection: {e}")
                 processed_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Show raw frame on error
                 success, norm_x, norm_y, pinch_state, dbl_click, scr_down, scr_up = False, 0.5, 0.5, 'no_hand', False, False, False


            # --- Update Shared State ---
            # This is where the main thread updates the data used by the mouse thread
            st.session_state.gesture_data['success'] = success
            st.session_state.gesture_data['x_index'] = norm_x
            st.session_state.gesture_data['y_index'] = norm_y
            st.session_state.gesture_data['pinch_state'] = pinch_state
            st.session_state.gesture_data['double_click'] = dbl_click
            st.session_state.gesture_data['scroll_down'] = scr_down
            st.session_state.gesture_data['scroll_up'] = scr_up

            # --- Update UI ---
            # Display the processed frame (with landmarks)
            frame_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)

            # Update status indicators
            pinch_icon = "🤏" if pinch_state == 'full_pinch' else ("🖐️" if pinch_state == 'half_pinch' else "❌")
            dbl_click_icon = "✅" if dbl_click else "❌"
            scr_down_icon = "⬇️" if scr_down else "❌"
            scr_up_icon = "⬆️" if scr_up else "❌"
            hand_detected_text = "Yes" if success else "No"

            status_placeholder.markdown(
                f"""
                **Hand Detected:** {hand_detected_text}
                **Pinch State:** {pinch_state.replace('_', ' ').title()} {pinch_icon}
                **Double Click:** {'On' if dbl_click else 'Off'} {dbl_click_icon}
                **Scroll Down:** {'On' if scr_down else 'Off'} {scr_down_icon}
                **Scroll Up:** {'On' if scr_up else 'Off'} {scr_up_icon}
                **Mouse Pos (Norm):** ({norm_x:.2f}, {norm_y:.2f})
                """
            )

            # Update 3D Box plot
            # Only update if rotation actually changed? Can cause flicker if updated too often.
            # Let's update every frame for simplicity, Plotly might optimize rendering.
            try:
                box_placeholder.plotly_chart(
                    create_3d_box(st.session_state.rotation['x'], st.session_state.rotation['y'], st.session_state.rotation['z']),
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error updating 3D Box: {e}")


            # Add a tiny sleep if Streamlit feels laggy, but usually not needed
            # time.sleep(0.01)

    elif not st.session_state.is_running:
        # Cleanup if stopped externally or on error
        if st.session_state.camera and st.session_state.camera.isOpened():
            st.session_state.camera.release()
            st.session_state.camera = None
        if st.session_state.control_thread and st.session_state.control_thread.is_alive():
             st.session_state.stop_event.set()
             st.session_state.control_thread.join(timeout=0.5)


if __name__ == "__main__":
    pinch_control_app()