import cv2
import numpy as np
import math

def nothing(x):
    pass

# --- CONFIGURATION ---
# Default HSV values for general skin color (Adjust via sliders if needed)
h_min, s_min, v_min = 0, 20, 70
h_max, s_max, v_max = 20, 255, 255

# Virtual Object Position (Center of the screen roughly)
OBJ_X, OBJ_Y = 320, 240
OBJ_RADIUS = 40

# Thresholds for distance (pixels)
DIST_WARNING = 200
DIST_DANGER = 100 + OBJ_RADIUS  # Touch boundary

# Initialize Camera
cap = cv2.VideoCapture(0)

# Create a window for calibration sliders
cv2.namedWindow("Calibration")
cv2.createTrackbar("Hue Min", "Calibration", h_min, 179, nothing)
cv2.createTrackbar("Sat Min", "Calibration", s_min, 255, nothing)
cv2.createTrackbar("Val Min", "Calibration", v_min, 255, nothing)
cv2.createTrackbar("Hue Max", "Calibration", h_max, 179, nothing)
cv2.createTrackbar("Sat Max", "Calibration", s_max, 255, nothing)
cv2.createTrackbar("Val Max", "Calibration", v_max, 255, nothing)

print("System Started. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for natural interaction (mirror view)
    frame = cv2.flip(frame, 1)
    
    # 1. PRE-PROCESSING
    # Convert BGR image to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get current positions of the trackbars
    h_min = cv2.getTrackbarPos("Hue Min", "Calibration")
    s_min = cv2.getTrackbarPos("Sat Min", "Calibration")
    v_min = cv2.getTrackbarPos("Val Min", "Calibration")
    h_max = cv2.getTrackbarPos("Hue Max", "Calibration")
    s_max = cv2.getTrackbarPos("Sat Max", "Calibration")
    v_max = cv2.getTrackbarPos("Val Max", "Calibration")

    # Create the mask
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Clean up the mask (remove noise) using Morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # 2. TRACKING (Find Contours)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Variables for logic
    current_state = "SAFE"
    color_status = (0, 255, 0) # Green
    distance = 1000 # Default large distance

    if contours:
        # Assume the largest contour is the hand
        max_contour = max(contours, key=cv2.contourArea)
        
        # Filter out small noise (must be bigger than 1000 pixels area)
        if cv2.contourArea(max_contour) > 1000:
            # Get the bounding box of the hand
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Calculate the "Centroid" (Center of the hand)
            cx = x + w // 2
            cy = y + h // 2

            # Draw the hand contour and center
            cv2.drawContours(frame, [max_contour], -1, (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

            # 3. LOGIC (Distance Calculation)
            # Euclidean distance between Hand Center and Virtual Object Center
            distance = math.sqrt((cx - OBJ_X)**2 + (cy - OBJ_Y)**2)
            
            # Draw line connecting hand to object
            cv2.line(frame, (cx, cy), (OBJ_X, OBJ_Y), (200, 200, 200), 2)

            # 4. STATE MACHINE
            if distance < DIST_DANGER:
                current_state = "DANGER"
                color_status = (0, 0, 255) # Red
            elif distance < DIST_WARNING:
                current_state = "WARNING"
                color_status = (0, 255, 255) # Yellow
            else:
                current_state = "SAFE"
                color_status = (0, 255, 0) # Green

    # 5. VISUALIZATION OVERLAY
    
    # Draw Virtual Object
    cv2.circle(frame, (OBJ_X, OBJ_Y), OBJ_RADIUS, color_status, 3)
    cv2.putText(frame, "TARGET", (OBJ_X - 25, OBJ_Y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_status, 2)

    # Draw Status Text
    cv2.putText(frame, f"STATE: {current_state}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 3)
    cv2.putText(frame, f"Dist: {int(distance)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Danger Overlay
    if current_state == "DANGER":
        cv2.putText(frame, "DANGER DANGER", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # Show the frames
    # Show the mask (useful for debugging/calibration)
    cv2.imshow("Mask (Calibration View)", mask) 
    # Show the main result
    cv2.imshow("Arvyax Assignment - Hand Track", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()