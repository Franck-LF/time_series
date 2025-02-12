

import pandas as pd
import cv2
import mediapipe as mp
import csv



def code_Bertrand(video_path):
    ''' Code provided by Bertrand - Display video '''

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





def get_landmarks(video_path, csv_name):
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

        # Crop and resize the image
        # frame = frame[70:, 200:-110]
        # frame = cv2.resize(frame, (0, 0), fx = 2.5, fy = 2.5)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # stretch_near = cv2.resize(frame, (780, 540), interpolation = cv2.INTER_LINEAR)

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
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    # Save landmarks in CS vfile
    df_landmarks = pd.DataFrame(csv_data, columns = ['frame_number', 'landmark_id', 'x', 'y', 'z'])
    df_landmarks.to_csv(csv_name, sep=',', index=False)
    return df_landmarks
    



def crop_video(video_path):
    ''' 
       src: https://medium.com/@sagarydv002/video-processing-with-opencv-cropping-and-resizing-window-51ca0eec43b7

    '''
    pass
    





# ------------------------- #
#                           #
#           MAIN            #
#                           #
# ------------------------- #


if __name__ == '__main__':

    video_path = "videos/seq_a.mov"
    # code_Bertrand(video_path)
    df_landmarks = get_landmarks(video_path, "landmarks.csv")



