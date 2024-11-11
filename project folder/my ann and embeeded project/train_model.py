import sys
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # Use tensorflow's keras utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import cv2
import numpy as np
import os
import mediapipe as mp

mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

# Set up data path and model parameters
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(sys.argv[1:])
no_sequences = 30
sequence_length = 30


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                 
    results = model.process(image)                
    image.flags.writeable = True                  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Left":
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def extract_right_hand_points(results):
    keypoints=np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Left":
                keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()              
    return keypoints
                
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) 
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
for action in actions:
    
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, hands)
                draw_landmarks(image,results)
                extract_right_hand_points(results)
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screenq
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                #keypoints extraction and saving
                keypoints = extract_right_hand_points(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)
                
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break


cap.release()
cv2.destroyAllWindows()


# Mapping labels to integers
label_map = {label: num for num, label in enumerate(actions)}
print("Label map:", label_map)


# Load sequences and labels
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"), allow_pickle=True)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
print("X shape:", X.shape)  # Debug print
y = to_categorical(labels).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
print("Training shape:", X_train.shape, "Testing shape:", X_test.shape)

# Build and compile model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))  

# Using categorical_crossentropy for multiclass (2 classes) setup
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# Callbacks for model performance and overfitting prevention
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
mc = ModelCheckpoint('best_model.keras', monitor='val_loss', mode='min', save_best_only=True)

# Train model with validation split
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[tb_callback, early_stopping, mc])

# Save model
model.save('Forward_recognition_model.h5')
model.summary()

#testing THE MODEL

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()


print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))
