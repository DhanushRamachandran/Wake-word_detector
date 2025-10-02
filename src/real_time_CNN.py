# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 00:10:06 2025
@author: sudha
"""


import sounddevice as sd
import numpy as np
import librosa
import threading
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from wake_word_config import all_configs
#import noisereduce as nr  # Optional


features_realtime = []
# Configs
sampling_rate = all_configs["sampling_configs"]["sampling_rate"]
sampling_rate = 44100
duration = all_configs["sampling_configs"]["time"]
duration = 2
n_mfcc = all_configs["feature_extraction"]["n_mfcc"]
fixed_T = 173  # adjust this based on your training data

# Load CNN model
model = load_model(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\model\Hey_mike_CNN_model32_800_samp.h5")

stop_flag = False

def record_audio():
    audio = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio) 


"""
data, sample_rate = librosa.load(audio_file, sr=None)

# Extract MFCCs
mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)

# Compute delta and delta-delta
delta = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)

# Stack them → (n_mfcc*3, T)
stacked_features = np.vstack([mfcc, delta, delta2])

# Do NOT take mean!
mfcc_features.append([stacked_features, class_label])

"""



def extract_features(audio):
    # Trim or pad audio to ensure consistent length
    sampling_rate = 44100
    expected_len = int(sampling_rate * duration) 
    print("--------exp length of audio: ",expected_len)
    print("-------actual len: ",len(audio))
    #if len(audio) < expected_len:
        
     #   audio = np.pad(audio, (0, expected_len - len(audio)), mode='constant')
    #else:
     #   audio = audio[:expected_len]
    
    # Extract MFCC + delta + delta-delta
    #data,sampling_rate = librosa.load(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack: shape (n_mfcc * 3, T)
    stacked = np.vstack([mfcc, delta, delta2])

    # Pad or truncate T to fixed_T
    if stacked.shape[1] < fixed_T:
        pad_width = fixed_T - stacked.shape[1]
        stacked = np.pad(stacked, ((0, 0), (0, pad_width)), mode='constant')
    else:
        stacked = stacked[:, :fixed_T]

    # Reshape for CNN: (features, time, 1)
    return stacked[..., np.newaxis]

def predict_wake_word(audio):
    features = extract_features(audio)
    print(features.shape)
    features = np.expand_dims(features, axis=0)  # shape: (1, n_features, T, 1)
    features_realtime.append(features)
    prediction = model.predict(features, verbose=0)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_label, confidence

def wake_word_listener():
    global stop_flag
    while not stop_flag:
        print("🎙️ Listening...")
        audio = record_audio()
        label, confidence = predict_wake_word(audio)
        print("label: ", label)
        print("confidence: ", confidence)
        if label == 1 and confidence > 0.6:
            print("✅ Wake word detected! (Confidence: {:.2f})".format(confidence))
        else:
            print("❌ No wake word. (Confidence: {:.2f})".format(confidence))
        time.sleep(0.1)

# Start listener thread
listener_thread = threading.Thread(target=wake_word_listener)
listener_thread.start()

# Wait for user to stop
try:
    while True:
        time.sleep(0.5)
except KeyboardInterrupt:
    stop_flag = True
    listener_thread.join()
    print("🔚 Stopped wake word detection.")


# df = pd.read_pickle(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\Data/sample_tesing.csv")
# features = df["feature"].values
# features = np.array([feature for feature in features])
# features = features[..., np.newaxis]
# for feature in features:
#     print(features.shape)
#     prediction = model.predict(featuress,verbose=0)
#     print(prediction)

for feature in features_realtime:
    pred = model.predict(feature)
    print("pred: ",pred)