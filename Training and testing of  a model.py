from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # Use tensorflow's keras utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score



import numpy as np
import os

# Set up data path and model parameters
DATA_PATH = os.path.join('MP_Data') 
actions = np.array(['forward', 'backward'])  # Assuming two actions
no_sequences = 30
sequence_length = 30

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
