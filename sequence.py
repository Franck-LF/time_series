# Faites des recherches sur le code ci-dessous et expliquez ce qu'il fait.

import cv2

video_path = "seq_a.mov"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()