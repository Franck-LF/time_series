

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


def keep_a_few_landmarks(df, landmarks_to_keep):
    ''' filter row of df to keep only a few landmarks '''
    df = df[df['landmark_id'].isin(landmarks_to_keep)]
    return df.sort_values(by = ['landmark_id', 'frame_number'])


def keep_one_landmark(df, landmark):
    ''' filter row of df to keep only one landmark '''
    df = df[df['landmark_id'] == landmark]
    return df.sort_values(by = 'frame_number')


def get_best_landmark_and_coord(df):
    '''
        Detect stationnarity in signal

        Return:
         - best_landmark: id of the landmark showing the longest stationnarity,
         - best_coord (string): 'x', 'y' or 'z',
         - best_first_frame (int): first frame of stationnarity,
         - best_last_frame (int):  last frame of stationnarity.
    '''
    assert False
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
            frame_end += window
            frame_start += 10
            frame_end   -= 10
            nb_frames = frame_end - frame_start + 1 

            if nb_frames > max_nb_frames:
                max_nb_frames = nb_frames
                best_landmark = landmark
                best_coord = coord
                best_first_frame = frame_start
                best_last_frame   = frame_end

    return best_landmark, best_coord, best_first_frame, best_last_frame


def get_longest_sationnarity_zone(df):
    ''' Detect stationnarity in signal '''
    max_nb_frames = 0
    best_coord    = ''
    best_first_frame = -1
    best_last_frame  = -1
    window   = 50

    for coord in ['x', 'y', 'z']:
        pvalues = []

        for i in range(df.shape[0] - window):
            res = adfuller(df[i:i+window:][coord].values)
            pvalues.append(res[1])

        mask = np.where(np.array(pvalues) < 0.045, 1, 0)
        frame_start, frame_end = get_max_range_of_ones(mask)
        frame_end   += window
        frame_start += 10
        frame_end   -= 10
        nb_frames = frame_end - frame_start + 1 

        if nb_frames > max_nb_frames:
            max_nb_frames    = nb_frames
            best_coord       = coord
            best_first_frame = frame_start
            best_last_frame  = frame_end

    return best_coord, best_first_frame, best_last_frame


def get_dic_frame_reps(first_frame, last_frame, nb_reps):
    ''' Compute a dictionary with the frame number and the number of the repetition to be displayed on the frame '''
    arr_frames_reps = np.linspace(first_frame, last_frame, nb_reps + 1).astype(int)
    dic_frame_reps = {arr_frames_reps[i]:str(i+1) for i in range(len(arr_frames_reps) - 1)}
    dic_frame_reps[arr_frames_reps[-1]] = ''
    return dic_frame_reps


def get_period(signal):
    ''' Detect the period of a signal

        Return: Integer (number of repetitions in the signal)

        Arg:
         - signal (numpy.array with floats): contains the signal.
    '''
    assert False
    # For each split of the signal,
    # we compute MAE, RMSE between different splits. 
    # (Measure of difference between superposition of two signals)

    metrics = []
    for n_split in range(1, 40):
        length = signal.shape[0] // n_split

        s = signal[0:length]
        mae, rmse = 0, 0

        for i in range(n_split - 1):
            s_ = signal[i*length : (i+1)*length]
            rmse += root_mean_squared_error(s, s_)
            mae += mean_absolute_error(s, s_)
        
        metrics.append([rmse, mae])
        # To Finish


def get_period_FFT(signal):
    ''' From DeepSeek (for explanations see file FFT.ipynb) '''
    fs = 1000     # Fréquence d'échantillonnage (Hz)
    T = 1.0 / fs  # Période d'échantillonnage

    # Calcul de la FFT
    N = len(signal)
    yf = fft(signal)
    xf = np.fft.fftfreq(N, T)[:N//2]  # Fréquences correspondantes

    # Trouver la fréquence dominante (en ignorant la composante DC)
    magnitude = 2.0/N * np.abs(yf[:N//2])
    dominant_frequency = xf[np.argmax(magnitude[1:]) + 1]  # Ignorer la composante DC

    # Calculer la période
    period = 1.0 / dominant_frequency
    return fs * period


def code_Bertrand(video_folder, video_name):
    ''' Code provided by Bertrand - Display video '''

    # Create an object "VideoCapture" with the video 
    cap = cv2.VideoCapture(video_folder + video_name)

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


def get_landmarks_from_video(video_folder, video_name):
    ''' Get landmarks of the video

        src: https://medium.com/@riddhisi238/real-time-pose-estimation-from-video-using-mediapipe-and-opencv-in-python-20f9f19c77a6

        return: dataframe with landmarks of the video
        Arg: video_folder (string): folder path of the video to analyse
    '''
    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_folder + video_name)

    frame_number = 0
    csv_data = []

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx = 2.0, fy = 2.0)
        result = pose.process(frame)

        # Draw the pose landmarks on the frame.
        if result.pose_landmarks:
            for idx, landmark in enumerate(result.pose_landmarks.landmark):
                csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
            frame_number += 1

    cap.release()
    # Save landmarks infos in csv file.
    df_landmarks = pd.DataFrame(csv_data, columns = ['frame_number', 'landmark_id', 'x', 'y', 'z'])
    df_landmarks.to_csv(video_folder + 'landmarks_' + video_name[:-4] + '.csv', sep=',', index=False)
    return df_landmarks


def display_video_and_landmarks(video_folder, video_name):
    ''' Display a video, get the landmarks and write them in a csv file

        src: https://medium.com/@riddhisi238/real-time-pose-estimation-from-video-using-mediapipe-and-opencv-in-python-20f9f19c77a6

        return: dataframe with landmarks of the video
        Args:
         - video_folder (string): folfer path of the video to analyse,
         - csv_name (string):   name of the file to save the data.
    '''
    assert(False)
    frames_to_display = [100, 126, 151, 177, 203, 228, 254, 280, 306, 331]

    # Initialize MediaPipe Pose and Drawing utilities
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose()

    # Open the video file
    cap = cv2.VideoCapture(video_folder + video_name)

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
    df_landmarks.to_csv(video_folder + 'landmarks_' + video_name[:-4] + '.csv', sep=',', index=False)
    return df_landmarks
    

def display_video_with_repetition_counter(video_folder, video_name, dic_frame_reps):
    ''' Display counter for each reps on the video
    
        Args:
         - video_folder (string): folder path of the video to analyse,
         - arr_frames (numpy.array): .
    '''
    # Open the video file
    cap = cv2.VideoCapture(video_folder + video_name)

    frame_number = 0
    bDisplay = False
    str_to_display = ''

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Crop and resize the image
        frame = cv2.resize(frame, (0, 0), fx = 2, fy = 2)

        # Draw
        if frame_number in dic_frame_reps.keys():
            bDisplay = dic_frame_reps[frame_number] != ''
            str_to_display = dic_frame_reps[frame_number]
            
        if bDisplay:
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, str_to_display, (80, 250), font, 7, (0,255,255), 5, cv2.LINE_AA)

        frame_number += 1

        # Display the frame
        cv2.imshow("MediaPipe Pose", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def read_video_and_display_with_repetition_counter(video_folder, video_name):
    ''' Analyse a video and display it with number of reps '''

    # Params
    b_compute_landmarks      = False
    b_compute_frame_counters = False

    df_landmarks   = None
    dic_frame_reps = {}

    if not(b_compute_frame_counters):
        frame_reps_file_name = 'frame_reps_' + video_name[:-4] + '.npy'
        # print(frame_reps_file_name)
        # print(os.listdir(video_folder))

        # Read frame_reps from file (if exists).
        if frame_reps_file_name in os.listdir(video_folder):
            print(f'Read file with the reps: {frame_reps_file_name}')
            with open(video_folder + frame_reps_file_name, 'rb') as f:
                dic_frame_reps = np.load(f, allow_pickle = True)[0]
                print('ok', type(dic_frame_reps))
                print('ok', dic_frame_reps)
                return

            display_video_with_repetition_counter(video_folder, video_name, dic_frame_reps)
            return

    landmarks_file_name = 'landmarks_' + video_name[:-4] + '.csv'

    # def read_landmarks_from_file(file_name):
    #     return pd.read_csv(video_folder + landmarks_file_name, sep = ',')

    if not(b_compute_landmarks) and (landmarks_file_name in os.listdir(video_folder)):
        # Read landmarks infos from file (if exists).
        print(f'Read file with the landmarks: {landmarks_file_name}')
        df_landmarks = pd.read_csv(video_folder + landmarks_file_name, sep = ',')

    else:
        # Get the landmarks from the video.
        df_landmarks = get_landmarks_from_video(video_folder, video_name)

    # Extract data for only one landmark.
    landmark = 'NOSE' # Should detect which landmark got the biggest variation / periodicity
    df_one_landmark = keep_one_landmark(df_landmarks, landmark)

    # ------------- #
    # Stationnarity #
    # ------------- #
    
    # Get first and last frames of the longest stationnarity zone.
    best_coord, first_frame, last_frame = get_longest_sationnarity_zone(df_one_landmark)

    # ------------- #
    #  Seasonnality #
    # ------------- #

    # Extract all frames in the "stationnarity zone"
    df_period = df_one_landmark[df_one_landmark['frame_number'].isin(range(first_frame, last_frame + 1))]

    # Get the period
    signal = df_period[best_coord].values
    n_period = get_period_FFT(signal)

    # Get number of repetitions in the signal
    nb_reps = int(signal.shape[0] / n_period)
    print("Nb reps:", nb_reps)

    # ----------------- #
    #  Display counter  #
    # ----------------- #
    dic_frame_reps = get_dic_frame_reps(first_frame, last_frame, nb_reps)
    with open(video_folder + 'frame_reps_' + video_name[:-4] + '.npy', 'wb') as f:
        np.save(f, dic_frame_reps)
    
    display_video_with_repetition_counter(video_folder, video_name, dic_frame_reps)
    






# ------------------------- #
#                           #
#           MAIN            #
#                           #
# ------------------------- #


if __name__ == '__main__':

    video_folder = "videos\\"
    video_name = "seq_a.mov"
    # video_name = "seq_b.mov"
    # video_name = "seq_tractions.mov"
    # video_name = "seq_multi_exos.mov"

    # code_Bertrand(video_folder, video_name)
    # get_landmarks_from_video(video_folder, video_name)
    # display_video_and_landmarks(video_folder, video_name)
    read_video_and_display_with_repetition_counter(video_folder, video_name)


