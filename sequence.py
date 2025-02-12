# Faites des recherches sur le code ci-dessous et expliquez ce qu'il fait.

import cv2

# Path of the video
video_path = "videos/seq_a.mov"

# Create an object "VideoCapture" with the video 
cap = cv2.VideoCapture(video_path)

# cap.isOpened(): Returns true if video capturing has been initialized already
while cap.isOpened():

    # cap.read(): Grabs, decodes and returns the next video frame.
    ret, frame = cap.read()

    # If there is no more frame BREAK THE LOOP
    if not ret:
        break

    # Display the frame
    cv2.imshow("Video", frame)

    # waitkey(delay): wait for 'delay' milisec AND
    # if a key is pressed during that time return the ASCII code of the key (we apply a mask)
    # otherwise return -1
    # 
    # the delay controls the speed of the video
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Probably clean the memory of "cap" object
cap.release()

# Straight forward
cv2.destroyAllWindows()