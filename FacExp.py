
"""
Facial Expression Detector - School Expo Demo
Uses webcam to detect faces and predict expressions.

Press:
  Q - Quit
  S - Save a photo with the detected expression
"""

import cv2
from fer.fer import FER

# Create emotion detector
emotion_detector = FER(mtcnn=True)

# Start webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam. Please check camera permissions.")
    exit()

print("Webcam started. Show different expressions!")
print("Press Q to quit, S to save a photo.")

photo_number = 1

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to read from webcam.")
        break

    # Flip for a natural mirror view
    frame = cv2.flip(frame, 1)

    # Detect faces and emotions
    results = emotion_detector.detect_emotions(frame)

    for result in results:
        # Face location
        x, y, w, h = result["box"]

        # Emotion scores
        emotions = result["emotions"]
        main_emotion = max(emotions, key=emotions.get)
        confidence = emotions[main_emotion] * 100

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display emotion
        label = f"{main_emotion.upper()} ({confidence:.1f}%)"
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # Title on screen
    cv2.putText(
        frame,
        "AI Facial Expression Detector",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imshow("Facial Expression Detector", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        filename = f"expression_{photo_number}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        photo_number += 1

camera.release()
cv2.destroyAllWindows()
