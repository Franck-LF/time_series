

import pandas as pd
import cv2
import mediapipe as mp
import csv



def test1(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
    # for i in range(10):
        if not(cap.isOpened()):
            break

        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def test2(video_path, csv_name):
    ''' 
       src: https://medium.com/@riddhisi238/real-time-pose-estimation-from-video-using-mediapipe-and-opencv-in-python-20f9f19c77a6

    '''

    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    csv_data = []

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        result = pose.process(frame_rgb)

        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Add the landmark coordinates to the list and print them
            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
            frame_number += 1

        # Display the frame
        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    # Save landmarks in CS vfile
    df = pd.DataFrame(csv_data, columns = ['frame_number', 'landmark id', 'x', 'y', 'z'])
    df.to_csv(csv_name)

    cap.release()
    cv2.destroyAllWindows()






# ------------------------- #
#                           #
#           MAIN            #
#                           #
# ------------------------- #


if __name__ == '__main__':

    video_path = "videos/seq_a.mov"
    # test1(video_path)
    test2(video_path, "landmarks.csv")



