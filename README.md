import cv2
import numpy as np

# Start webcam
cap = cv2.VideoCapture(0)

# Counter variables
count = 0
object_inside = False

# Define ROI (rectangle area)
x1, y1, x2, y2 = 200, 150, 450, 350

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame (mirror view)
    frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -----------------------------
    # COLOR DETECTION (RED)
    # -----------------------------
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = mask1 + mask2

    # Reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detected_inside_roi = False

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter small noise
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Center point
            cx = x + w // 2
            cy = y + h // 2

            # Check if inside ROI
            if x1 < cx < x2 and y1 < cy < y2:
                detected_inside_roi = True

    # -----------------------------
    # TRIGGER LOGIC (ENTER + EXIT)
    # -----------------------------
    if detected_inside_roi and not object_inside:
        object_inside = True

    elif not detected_inside_roi and object_inside:
        count += 1
        object_inside = False

    # -----------------------------
    # DRAW ROI
    # -----------------------------
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display text
    cv2.putText(frame, f"Objects Counted: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show webcam feed
    cv2.imshow("Object Counter System", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
