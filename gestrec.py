import joblib
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Process, Value
import ctypes
import sys
from time import sleep
from commandexec import COMMAND_WALK, COMMAND_WIGGLE, COMMAND_DOWN, COMMAND_SPIN, execute_command

from auxillary_funcs.ml import get_nearest_hand, preprocess_landmarks, write_csv, read_labels, bounding_rect
from auxillary_funcs.interprocess_comms import get_conn_listener, get_conn_client, GESTREC_PORT
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

_GESTREC_ON = 'on'
_GESTREC_OFF = 'off'
_GSTREC_STOP = 'stop'

_COMMANDS_DICT = {
    1: COMMAND_WIGGLE,
    2: COMMAND_WALK,
    3: COMMAND_DOWN,
    4: COMMAND_SPIN
}


class Gestrec():

    # aiding functions
    def __init__(self) -> None:                         # __init__: set member-variables
        # mediapipe
        self.mediapipe_model_complexity = 0
        self.mediapipe_min_detection_confidence = 0.8
        self.mediapipe_min_tracking_confidence = 0.8

        # model
        self.model_min_confidence = 0.9
        self.model_path = './model/logmod.pkl'
        self.model_labels_path = './data/labels.txt'
        self._model_active = Value(ctypes.c_bool, False)

        # development mode
        self.dev_data_path = 'data/data.csv'
        self.dev_mode = False

        # openCV
        self.cv_cap_flip = True
        self.cv_cap_source = 0  # 0 == default device
        #self.cv_cap_source = 'http://192.168.123.12:8080/?action=stream'  # 0 == default device

        # processes
        self._cap_proc = None

        # private
        self.__cap = 0
        self.__hands = 0
        self.__land_q = 0
        self.__model = 0
        self.__labels = 0
        self.__label = 0
        self.__label_count = 0

    def __start_listener__(self):
        print("Entered __start_listener__")
        print("Listening to GESTREC_PORT")
        # initialize and start self.__run_gestrec__ as parallel process (multiprocessing)
        self._cap_proc = Process(target=self.__run_gestrec__)
        self._cap_proc.start()
        print("Started parallel process __run_gestrec__")
        # __run_gestrec__ listens to GESTREC_PORT = 4444
        listener = get_conn_listener(GESTREC_PORT)
        # polls permanently: process messages and set flags
        while True:
            con = listener.accept()
            msg = con.recv()                # received message on GESTREC_PORT = 4444
            print("received message on GESTREC_PORT = 4444")
            if msg == _GSTREC_STOP:
                self._cap_proc.terminate()
                sys.exit()
            elif msg == _GESTREC_OFF:
                print("received _GESTREC_OFF")
                self._model_active.value = False
            elif msg == _GESTREC_ON:
                print("received _GESTREC_ON")
                self._model_active.value = True

    def start(self):
        # initialize and start parallel process (multiprocessing)
        # starts the listener in parallel
        print("Entered gestrec.start()")
        proc = Process(target=self.__start_listener__)
        proc.start()
        sleep(1)
        return proc


    def process_gesture(self, image):
        #print("Got into classification block")
        #sleep(10)
        predict_result = np.squeeze(self.__model.predict_proba(np.array(self.__land_q).reshape(1, -1)))       # pass the queue
        idx = np.argmax(predict_result)
        gesture, confidence = self.__labels[idx], predict_result[idx]
        if idx != 0:
            if confidence >= self.model_min_confidence:
                # execute command
                # deactivate for work with webcam
                # execute_command(_COMMANDS_DICT[idx])
                self.__land_q.clear()
            # print text to image
            cv2.putText(
                image, f'Gesture: {gesture}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0),
                4, cv2.LINE_AA)
            cv2.putText(
                image, f'Gesture: {gesture}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255),
                2, cv2.LINE_AA)
            cv2.putText(image, f'Confidence: {confidence:.3f}', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, f'Confidence: {confidence:.3f}', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


    def process_landmarks(self, image, results):
        # process hand landmarks
        hand_landmarks = get_nearest_hand(results.multi_hand_landmarks)
        landmarks = preprocess_landmarks(hand_landmarks)
        self.__land_q.append(landmarks)        # queue

        print("Added to queue")

        # process landmarks and save to CSV when in dev_mode
        if self.dev_mode and self.__label != -1:
            if len(self.__land_q) == self.__land_q.maxlen:
                write_csv(self.__label, list(self.__land_q), self.dev_data_path)
                self.__land_q.clear()
                print(f'done_saving {self.__label_count}st label with id {self.__label}')
                self.__label_count += 1
                self.__label = -1

        # predict gesture using model
        #print(self.dev_mode,self._model_active.value, len(land_q))
        if not self.dev_mode and self._model_active.value and len(self.__land_q) == self.__land_q.maxlen:
            self.process_gesture(image)

        # draw boundingbox
        x, y, w, h = bounding_rect(image, hand_landmarks)
        cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 1)
        cv2.rectangle(image, (x, y), (w, y-22), (0, 0, 255), -1)
        cv2.putText(image, 'Tracking', (x + 5, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # draw hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())



    # runs in parallel
    # main gestrec process
    def __run_gestrec__(self):
        # video capture ################################################################################################
        self.__cap = cv2.VideoCapture(self.cv_cap_source)

        # mediapipe ####################################################################################################
        self.__hands = mp_hands.Hands(
            model_complexity=self.mediapipe_model_complexity,
            min_detection_confidence=self.mediapipe_min_detection_confidence,
            min_tracking_confidence=self.mediapipe_min_tracking_confidence,
            max_num_hands=100,
        )

        # landmarks queue ##############################################################################################
        self.__land_q = deque(maxlen=32)

        # ml model #####################################################################################################
        self.__model = joblib.load(self.model_path) if not self.dev_mode else None
        self.__labels = read_labels(self.model_labels_path) if not self.dev_mode else None

        # app ##########################################################################################################
        self.__label = -1
        self.__label_count = 1

        print("Prepared __run_gestrec__")
        # runs in parallel as long as cap.isOpened is true
        while self.__cap.isOpened():
            # capture image
            success, image = self.__cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            if self.cv_cap_flip:
                image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # process with mediapipe
            results = self.__hands.process(image)

            # revert image color back
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # capture key imput
            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # esc
                cv2.destroyAllWindows()
                self.__cap.release()
                break
            if 48 <= key <= 57:  # 0 - 9
                self.__label = key - 48

            if results.multi_hand_landmarks:
                #process_Landmarks
                self.process_landmarks(image, results)


            # show capture
            cv2.imshow('MediaPipe Hands', image)


def __send_command__(command):              # MP-command e.g. _GESTREC_ON
    con = get_conn_client(GESTREC_PORT)     # return MP-Client((ADDRESS, port), authkey=AUTHKEY)
    con.send(command)
    con.close()


def gestrec_stop():
    __send_command__(_GSTREC_STOP)


def gestrec_on():
    __send_command__(_GESTREC_ON)


def gestrec_off():
    __send_command__(_GESTREC_OFF)
