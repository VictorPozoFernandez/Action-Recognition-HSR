import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import mediapipe as mp
import os
from geometry_msgs.msg import PoseStamped
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.callbacks import TensorBoard
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# PENDIENTE:  Desplazar 1 frame al analizar una nueva sequencia en vez de borrar la lista por completo,
# Crear Github respository, 
# Corregir full path
# Guardar secuencias enteras en vez de frames
# Testear el modelo usando X_test Y_test, 
# Comparar % aciertos usando mas o menos sequencias, o mas o menos ephocs de entrenamiento. 


# PARAMETERS
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
DATA_PATH = os.path.join(PATH,"MP_Data")
MODEL_PATH = os.path.join(PATH,"action.h5")
actions = np.array(os.listdir(DATA_PATH))
actions_to_record = np.array(["hello"])
threshold = 0.9
simulation = False
train = False
num_sequences = 30
num_frames_sequence = 30

def listen(model):

    rospy.init_node("action_recognizer", anonymous=True)
    rospy.loginfo("Node action_recognizer initialized. Listening...")
    
    if simulation == True:
        sequence = []
        sentence = []
        #rospy.Subscriber("/usb_cam/image_raw", Image, callback, (sequence, sentence))
        rospy.Subscriber("/hsrb/head_rgbd_sensor/rgb/image_rect_color", Image, callback, (sequence, sentence))
    else:
        VideoCapture(model)

    rospy.spin()
    cv2.destroyAllWindows()


def callback(img_msg, args):

    sequence = args[0]
    sentence = args[1]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        bridge = CvBridge()
        frame= bridge.imgmsg_to_cv2(img_msg)
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        cv2.imshow('OpenCV Feed', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            sequence = []

            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                        print(actions[np.argmax(res)])
                        speak(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
                    print(actions[np.argmax(res)])
                    speak(actions[np.argmax(res)])
        

def VideoCapture(model):

    sequence = []
    sentence = []
    
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
                if res[np.argmax(res)] > threshold: 
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

    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size= 10)
    rospy.Rate(1)

    if msg == "coffee":
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
        for sequence in range(num_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

    cap = cv2.VideoCapture(0) 

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        for action in actions_to_record:
            for sequence in range(num_sequences):
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
                    np_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(np_path, keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                        
        cap.release()
        cv2.destroyAllWindows()


def obtain_model(actions, train = False):
    label_map = {label:num for num, label in enumerate(actions)}

    sequences, labels = [], []
    for action in actions:
        for sequence in range(num_sequences):
            seq = []
            for frame_num in range(num_frames_sequence):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                seq.append(res)
            sequences.append(seq)
            labels.append(label_map[action])

    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    if train:
        model.fit(X_train, Y_train, epochs=100, callbacks=[tb_callback])
        model.save(MODEL_PATH)
    else:
        model.load_weights(MODEL_PATH)
        return model




if __name__ == '__main__':
    try:
        
        if train:
            collect_datapoints(actions_to_record)
            obtain_model(actions, train)
        else:
            model = obtain_model(actions, train)
            listen(model)

    except rospy.ROSInterruptException:
        pass