import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# Initialize MediaPipe Holistic (Pose + Hand tracking)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Model Settings: Optimized for video speed vs accuracy
holistic = mp_holistic.Holistic(
    static_image_mode=False,       # False = video mode (uses tracking)
    model_complexity=1,            # 1 = balanced (0=fast, 2=accurate)
    smooth_landmarks=True,         # Reduces jitter
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Signal Smoothing Buffers (prevents flickering text)
# We store the last 5 frames of data and take the most common result
BUFFER_SIZE = 5
left_signal_buffer = deque(maxlen=BUFFER_SIZE)
right_signal_buffer = deque(maxlen=BUFFER_SIZE)

# ==========================================
# 2. GEOMETRY HELPER FUNCTIONS
# ==========================================

def calculate_angle(a, b, c):
    """
    Calculates the inner angle of the elbow (0-180 degrees).
    Args: a (Shoulder), b (Elbow), c (Wrist)
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def calculate_inclination(p1, p2):
    """
    Calculates the angle of a segment relative to the screen's horizontal axis.
    0 deg = Right, 180 deg = Left, 90 deg = Down, -90 deg = Up.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.degrees(np.arctan2(dy, dx))

def is_palm_open(hand_landmarks):
    """
    Returns True if at least 3 fingers are extended.
    Logic: Finger Tip is further from wrist than the PIP joint.
    """
    if not hand_landmarks: return False
    wrist = hand_landmarks.landmark[0]
    fingers_open = 0
    
    # Indices for Tip (8,12,16,20) and PIP Joint (6,10,14,18) - exclude Thumb
    for tip_idx, pip_idx in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        
        # Euclidean distance check
        dist_tip = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        dist_pip = ((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)**0.5
        
        # If tip is 1.2x further than joint, it's open
        if dist_tip > dist_pip * 1.2:
            fingers_open += 1
            
    return fingers_open >= 3

def get_palm_orientation(hand_landmarks, side):
    """
    Uses cross-product of palm vectors to determine if palm faces Down/Up/Vertical.
    Crucial for distinguishing 'Slow Down' (Palm Down).
    """
    if not hand_landmarks: return 'UNKNOWN'
    
    p0 = hand_landmarks.landmark[0]   # Wrist
    p5 = hand_landmarks.landmark[5]   # Index Base
    p17 = hand_landmarks.landmark[17] # Pinky Base
    
    # Vector 1: Wrist -> Index; Vector 2: Wrist -> Pinky
    vec_index = np.array([p5.x - p0.x, p5.y - p0.y, p5.z - p0.z])
    vec_pinky = np.array([p17.x - p0.x, p17.y - p0.y, p17.z - p0.z])
    
    # Normal vector perpendicular to palm
    normal = np.cross(vec_index, vec_pinky)
    norm_mag = np.linalg.norm(normal)
    if norm_mag == 0: return 'UNKNOWN'
    
    y_comp = (normal / norm_mag)[1] # Y-component determines Up/Down tilt
    
    if abs(y_comp) < 0.4: return 'VERTICAL'
    
    # Left/Right hands have opposite cross-product outcomes
    if side == 'Left':
        return 'DOWN' if y_comp < 0 else 'UP'
    else: # Right
        return 'DOWN' if y_comp > 0 else 'UP'

def get_most_common(buffer):
    """Smoothing function to return the most frequent signal in the buffer."""
    if not buffer: return "CRUISING"
    return max(set(buffer), key=buffer.count)

# ==========================================
# 3. CORE SIGNAL LOGIC (RELAXED)
# ==========================================

def get_arm_signal(shoulder, elbow, wrist, hand_landmarks, side):
    """
    Determines the signal for a specific arm based on angles and hand state.
    Includes relaxed tolerances for real-world usage.
    """
    # --- A. Calculate Angles ---
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    upper_arm_inc = calculate_inclination(shoulder, elbow) # Shoulder to Elbow
    forearm_inc = calculate_inclination(elbow, wrist)      # Elbow to Wrist
    whole_arm_inc = calculate_inclination(shoulder, wrist) # Shoulder to Wrist (Direct)

    # --- B. Define Recognition Zones (Relaxed) ---
    
    # 1. Upper Arm Horizontal Check
    # Allow elbow to droop up to 35 degrees below shoulder level
    angle_tolerance = 35 
    if side == 'Left':
        # Left arm extends to ~180 degrees
        is_upper_horizontal = (abs(upper_arm_inc) > (180 - angle_tolerance))
    else:
        # Right arm extends to ~0 degrees
        is_upper_horizontal = (abs(upper_arm_inc) < angle_tolerance)

    # 2. Forearm Vertical Check
    # Allow arm to lean forward/back by 40 degrees
    forearm_tolerance = 40
    # -90 is straight UP, 90 is straight DOWN
    is_forearm_up = (abs(forearm_inc - (-90)) < forearm_tolerance)
    is_forearm_down = (abs(forearm_inc - 90) < forearm_tolerance)

    # --- C. Logic Decision Tree ---
    
    # High Arm Signals (Turn or Stop)
    if is_upper_horizontal:
        
        # 1. STRAIGHT ARM TURN
        # If elbow angle is large (> 140), it's a straight arm signal
        if elbow_angle > 140:
            return "LEFT TURN" if side == 'Left' else "RIGHT TURN"
            
        # 2. BENT ELBOW SIGNALS (Box Shape)
        # Accept a wide range of bends (60 to 140 degrees)
        elif 60 < elbow_angle <= 140:
            
            if is_forearm_up:
                # OPPOSITE TURN LOGIC
                # Left Arm Up = Right Turn | Right Arm Up = Left Turn
                return "RIGHT TURN" if side == 'Left' else "LEFT TURN"
            
            if is_forearm_down:
                # STOP LOGIC
                return "STOP"

    # Low Arm Signal (Slow Down)
    # Logic: Arm extended downwards but slightly away from body (15-45 degrees)
    angle_off_vertical = abs(whole_arm_inc - 90)
    is_in_slow_down_zone = 15 < angle_off_vertical < 45
    
    if is_in_slow_down_zone:
        # Must verify hand state (Palm must be Open and Facing Down)
        palm_open = is_palm_open(hand_landmarks)
        palm_dir = get_palm_orientation(hand_landmarks, side)
        
        if palm_open and palm_dir in ['DOWN', 'VERTICAL']:
            return "SLOW DOWN"

    return "CRUISING"

# ==========================================
# 4. MAIN EXECUTION LOOP
# ==========================================

def main():
    cap = cv2.VideoCapture(0)

    # Window Setup
    cv2.namedWindow('Cyclist Signal Detector', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cyclist Signal Detector', 1000, 600)

    print("--- CYCLIST SIGNAL DETECTOR ACTIVE ---")
    print("1. Straight Arm: Turn in that direction.")
    print("2. Bent Arm Up: Turn in OPPOSITE direction.")
    print("3. Bent Arm Down: STOP.")
    print("4. Low Arm + Open Palm: SLOW DOWN.")
    print("Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success: continue

        # 1. Mirror the image (Selfie View logic)
        image = cv2.flip(image, 1)
        
        # 2. MediaPipe Processing
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # 3. Prepare for Drawing
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        current_left = "CRUISING"
        current_right = "CRUISING"
        active_signal = "CRUISING"
        status_color = (200, 200, 200) # Default Gray

        # 4. Extract Logic if Pose Detected
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            h, w, c = image.shape
            
            # Coordinates for Left (11,13,15) and Right (12,14,16)
            l_sh, l_el, l_wr = [lm[11].x, lm[11].y], [lm[13].x, lm[13].y], [lm[15].x, lm[15].y]
            r_sh, r_el, r_wr = [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y]
            
            # Check visibility to avoid noise when arm is off-screen
            l_vis = lm[11].visibility > 0.6 and lm[15].visibility > 0.6
            r_vis = lm[12].visibility > 0.6 and lm[16].visibility > 0.6

            # Calculate raw signals
            if l_vis:
                current_left = get_arm_signal(l_sh, l_el, l_wr, results.left_hand_landmarks, 'Left')
            if r_vis:
                current_right = get_arm_signal(r_sh, r_el, r_wr, results.right_hand_landmarks, 'Right')

            # Buffer / Smoothing
            left_signal_buffer.append(current_left)
            right_signal_buffer.append(current_right)
            smooth_left = get_most_common(left_signal_buffer)
            smooth_right = get_most_common(right_signal_buffer)

            # Prioritize which signal to show (Left Arm > Right Arm usually)
            final_signal = "CRUISING"
            if smooth_left != "CRUISING":
                final_signal = smooth_left
                side_prefix = "LEFT ARM: "
            elif smooth_right != "CRUISING":
                final_signal = smooth_right
                side_prefix = "RIGHT ARM: "
            else:
                side_prefix = ""

            # Determine UI Color
            if "STOP" in final_signal:
                status_color = (0, 0, 255)      # Red
            elif "TURN" in final_signal:
                status_color = (0, 255, 255)    # Yellow
            elif "SLOW" in final_signal:
                status_color = (255, 100, 0)    # Blue/Orange
            
            active_signal = f"{side_prefix}{final_signal}"

            # Draw Skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 5. Draw UI Overlay
        # Background Box for Text
        (text_w, text_h), _ = cv2.getTextSize(active_signal, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(image, (0, 0), (text_w + 40, 80), (0, 0, 0), -1)
        
        # Text Signal
        cv2.putText(image, active_signal, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)

        cv2.imshow('Cyclist Signal Detector', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
