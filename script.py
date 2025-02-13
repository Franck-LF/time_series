

import os
import cv2
import PIL
import math
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt

# Scipy
from scipy import stats
from scipy.fftpack import fft, ifft
from scipy.stats import kruskal

# Scikit-Learn
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

# statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import seasonal_decompose




# -------------------------------
#
#       TOOLS
#
# -------------------------------

landmarks_to_keep = ['NOSE', 'MOUTH_LEFT', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                    'LEFT_WRIST','RIGHT_WRIST', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE',
                    'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
landmarks_to_keep.sort()




def get_max_range_of_ones(arr):
    ''' Return start and end indices of maximum continuous sequence of 1 in the array '''
    x_best, y_best = -1, -1
    max_counter = 0

    x, y = -1, -1
    counter = 0
    cur_value = arr[0]
    
    if cur_value:
        counter = 1
        x, y = 0, 0

    for i in range(1, len(arr) + 1):
        if i == len(arr):
            value = 0
        else:
            value = arr[i]

        if value == cur_value:
            if value:
                counter += 1
                y += 1
        else:
            if value: # back to 1
                x, y = i, i
                counter = 1
            else: # back to 0
                if counter > max_counter:
                    max_counter = counter
                    x_best, y_best = x, y
                counter = 0
        cur_value = value

    return x_best, y_best


def keep_a_few_landmarks(df):
    ''' filter row of df to keep only a few landmarks '''
    return df[df['landmark_id'].isin(landmarks_to_keep)]

def get_best_landmark_and_coord(df):
    '''
        Return:
         - best_landmark: id of the landmark showing the longest stationnarity,
         - best_coord (string): 'x', 'y' or 'z',
         - best_first_frame (int): 
         - best_last_frame (int): 
    '''
    max_nb_frames = 0
    best_landmark = -1
    best_coord    = ''
    best_first_frame = -1
    best_last_frame  = -1

    for landmark in range(20):
        window   = 50
        df_temp = df[landmark::20]
        print(landmarks_to_keep[landmark])

        for coord in ['x', 'y', 'z']:
            pvalues = []

            for i in range(df_temp.shape[0] - window):
                res = adfuller(df_temp[i:i+window:][coord].values)
                pvalues.append(res[1])

            mask = np.where(np.array(pvalues) < 0.045, 1, 0)
            frame_start, frame_end = get_max_range_of_ones(mask)
            # print(frame_start, frame_end)
            frame_end += window
            frame_start += 10
            frame_end   -= 10
            # print(frame_start, frame_end)
            nb_frames = frame_end - frame_start + 1 
            print('nb:', nb_frames)

            if nb_frames > max_nb_frames:
                max_nb_frames = nb_frames
                best_landmark = landmark
                best_coord = coord
                best_first_frame = frame_start
                best_last_frame   = frame_end

    return best_landmark, best_coord, best_first_frame, best_last_frame





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



def get_landmarks(video_path):
    ''' Get landmarks of the video

        src: https://medium.com/@riddhisi238/real-time-pose-estimation-from-video-using-mediapipe-and-opencv-in-python-20f9f19c77a6

        return: dataframe with landmarks of the video
        Arg: video_path (string): path of the video to analyse
    '''
    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    csv_data = []

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx = 2.0, fy = 2.0)
        result = pose.process(frame)

        # Draw the pose landmarks on the frame
        if result.pose_landmarks:
            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
            frame_number += 1

    cap.release()
    return df_landmarks
    




def display_video_with_landmarks(video_path, csv_name):
    ''' Display a video, get the landmarks and write them in a csv file

        src: https://medium.com/@riddhisi238/real-time-pose-estimation-from-video-using-mediapipe-and-opencv-in-python-20f9f19c77a6

        return: dataframe with landmarks of the video
        Args:
         - video_path (string): path of the video to analyse,
         - csv_name (string):   name of the file to save the data.
    '''
    frames_to_display = [100, 126, 151, 177, 203, 228, 254, 280, 306, 331]

    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_number = 0
    counter = 0
    to_be_displayed = 1
    csv_data = []
    bDisplay = False

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Crop and resize the image
        # frame = frame[70:, 200:-110]
        frame = cv2.resize(frame, (0, 0), fx = 2, fy = 2)

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Draw
        if (frame_number-1) in frames_to_display:
            bDisplay = True
            counter = 0
        
        if bDisplay:
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, str(to_be_displayed), (80, 250), font, 7, (0,255,255), 5, cv2.LINE_AA)
            counter += 1
            if counter == 3:
                bDisplay = False
                counter = 0
                to_be_displayed += 1

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
    



def display_video_with_counter(video_path):
    ''' Display video with number of reps '''
    df_landmarks = get_landmarks(video_path)

    df = df[df["landmark_id"] == "NOSE"]
    df = df[df['frame_number'].isin(range(frame_start, frame_end + 1))]
    pass





# ------------------------- #
#                           #
#           MAIN            #
#                           #
# ------------------------- #


if __name__ == '__main__':

    video_path = "videos/seq_a.mov"

    # code_Bertrand(video_path)
    # df_landmarks = get_landmarks(video_path)
    df_landmarks = display_video_with_landmarks(video_path, "landmarks.csv")
