import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import mediapipe as mp
import os
import shutil
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard
import keras
import time
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
DATA_PATH = os.path.join(PATH,"MP_Data")
MODEL_PATH = os.path.join(PATH,"action4.h5") #To use action3 model remember to cut the none1

# PARAMETERS
simulation = True
train = False
num_sequences = 90
num_frames_sequence = 30
threshold = 0.999

def listen(model):

    rospy.init_node("action_recognizer", anonymous=True)
    rospy.loginfo("Node action_recognizer initialized. Listening...")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
        if simulation == True:
            sequence = []
            sentence = []
            rospy.Subscriber("/usb_cam/image_raw", Image, callback, (sequence, sentence, holistic))
            #rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, callback, (sequence, sentence))
            current_pos = Pose()
            saved_pos = Pose()
            rospy.Subscriber("/hsrb/odom_ground_truth", Odometry, callback_odom)
            rospy.Publisher('/task_done', String, queue_size= 1)
            rospy.Subscriber("/task_done", String, callback_task_done)
        else:
            VideoCapture(model)

        rospy.spin()
        cv2.destroyAllWindows()

def callback_odom(msg):
    global current_pos 
    current_pos = msg.pose.pose

def callback(img_msg, args):

    sequence = args[0]
    sentence = args[1]
    holistic = args[2]
    actions = np.array(os.listdir(DATA_PATH))

    bridge = CvBridge()
    frame= bridge.imgmsg_to_cv2(img_msg)
    image, results = mediapipe_detection(frame, holistic)
    draw_styled_landmarks(image, results)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_msg = bridge.cv2_to_imgmsg(image, encoding="passthrough")
    pub = rospy.Publisher('/MP_image', Image, queue_size= 30)
    pub.publish(img_msg)
    

    # cv2.imshow('OpenCV Feed', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(1)
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        sequence = []

        if res[np.argmax(res)] >= threshold: 
            if len(sentence) > 0: 
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                    if actions[np.argmax(res)] != "none1":
                        print(actions[np.argmax(res)])
                    speak(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])
                if actions[np.argmax(res)] != "none1":
                    print(actions[np.argmax(res)])
                speak(actions[np.argmax(res)])
    

def VideoCapture(model):

    sequence = []
    sentence = []
    actions = np.array(os.listdir(DATA_PATH))
    
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            keypoints = extract_keypoints(results)

            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                sequence = []
                print(res)
                if res[np.argmax(res)] >= threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            print(actions[np.argmax(res)])
                            speak(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                        print(actions[np.argmax(res)])
                        speak(actions[np.argmax(res)])
        
        cap.release()
        cv2.destroyAllWindows()


def speak(msg):

    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size= 1)
    rospy.Rate(1)

    global current_pos
    global saved_pos

    if msg == "coffee":
        saved_pos = current_pos

        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = float(2.3)
        goal.pose.position.y = float(4.1)
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = float(0.7)
        goal.pose.orientation.w = float(0.7)

        pub.publish(goal)
        rospy.Rate(1)
    
    elif msg == "human":
        saved_pos = current_pos

        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = float(0.6)
        goal.pose.position.y = float(3.45)
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = float(-1)
        goal.pose.orientation.w = float(0)

        pub.publish(goal)
        rospy.Rate(1)

    else:
        pass

def callback_task_done(msg):
    global saved_pos

    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size= 1)
    rospy.Rate(1)

    if msg.data == "done":

        print("Going home")
        goal = PoseStamped()

        goal.header.seq = 1
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.pose.position.x = float(saved_pos.position.x + 2.42)
        goal.pose.position.y = float(saved_pos.position.y + 3.44)
        goal.pose.position.z = 0.0

        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = float(saved_pos.orientation.z + 0.62)
        goal.pose.orientation.w = float(saved_pos.orientation.w - 0.26)

        pub.publish(goal)
        rospy.Rate(1)

def mediapipe_detection(image, model):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) 
 
def extract_keypoints(results):

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])



def collect_datapoints(actions_to_record):


    for action in actions_to_record:
        try:
            os.makedirs(os.path.join(DATA_PATH, action))
        except:
            pass

    cap = cv2.VideoCapture(0) 
    rospy.init_node("collecting_Datapoints", anonymous=True)
    rate = rospy.Rate(12)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in actions_to_record:
            for sequence in range(num_sequences):
                seq = []
                for frame_num in range(num_frames_sequence):

                    ret, frame = cap.read()
            
                    if frame_num == 0: 
                        cv2.putText(frame, 'New recording...', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', frame)
                        cv2.waitKey(2000)

                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)
                    cv2.putText(image, 'Action: {}  Sequence number: {}'.format(action, sequence), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    
                    keypoints = extract_keypoints(results)
                    seq.append(keypoints)
                    rate.sleep()

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                seq = np.array(seq)
                np_path = os.path.join(DATA_PATH, action, str(sequence))
                np.save(np_path, seq)
    
                        
        cap.release()
        cv2.destroyAllWindows()


def obtain_model(actions, train = False):
    
    log_dir = os.path.join('Logs','action4')
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    if train:

        label_map = {label:num for num, label in enumerate(actions)}
        sequences, labels = [], []
        
        for action in actions:
            for sequence in range(num_sequences):
                seq = np.load(os.path.join(DATA_PATH, action, "{}.npy".format(sequence)))
                sequences.append(seq)
                labels.append(label_map[action])

        X = np.array(sequences)
        Y = to_categorical(labels).astype(int)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        model.fit(X_train, Y_train, epochs=300, callbacks=[tb_callback], validation_data=(X_test, Y_test)) 
        #model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))
        model.save(MODEL_PATH)
        
    else:
        model.load_weights(MODEL_PATH)
        return model




if __name__ == '__main__':
    try:
        
        if train:

            print('Enter the name of the new actions to record (separated by spaces):')
            x = input()

            if x == '':
                print('No actions entered. Training the model with the already registered actions:')
    
            action_list = x.split()
            actions_to_record = np.array(action_list)

            for count, action in enumerate(actions_to_record):
                if os.path.exists(os.path.join(DATA_PATH, action)):

                    t = True
                    while t == True:
                        print("The action {} already exists. Do you want to overwrite it's content? (y/n) ".format(action))
                        x = input()
                        if x == 'y' or x == 'n':
                            if x == 'y':
                                shutil.rmtree(os.path.join(DATA_PATH, action), ignore_errors=True)
                            if x == 'n':
                                actions_to_record = np.delete(actions_to_record, count)
                            t = False   
                        else:
                            print("Invalid return.")

            if actions_to_record.shape == (0,):
                actions = np.array(os.listdir(DATA_PATH))
                obtain_model(actions, train)

            else:        
                collect_datapoints(actions_to_record)
                actions = np.array(os.listdir(DATA_PATH))
                obtain_model(actions, train)

        else:
            actions = np.array(os.listdir(DATA_PATH))
            model = obtain_model(actions, train)
            listen(model)

    except rospy.ROSInterruptException:
        pass