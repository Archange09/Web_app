import cv2
from ultralytics import YOLO

# Load your custom YOLOv8 model
model = YOLO("USE_THIS.pt")  # Replace with your model's path

# Open the webcam (0 = default webcam, change to 1 or 2 if needed)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run inference using the custom model
    results = model(frame)

    # Get the annotated frame with detections
    annotated_frame = results[0].plot()

    # Display the annotated frame in a window
    cv2.imshow(f"Aug_ZJU", annotated_frame)

    # Press 'q' to exit the loop or close the window
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Custom Model Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
