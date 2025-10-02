# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 14:29:09 2025

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
import noisereduce as nr



sampling_rate = all_configs["sampling_configs"]["sampling_rate"]
duration = all_configs["sampling_configs"]["time"]
duration = 2
n_mfcc = all_configs["feature_extraction"]["n_mfcc"] # no of features for real time extraction
model = load_model(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\model\Hey_mike_trained_model_40_features_similar.h5")


stop_flag = False
def record_audio():
    audio = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=1, dtype='float32')
    sd.wait()
    #audio = nr.reduce_noise(y=audio, sr=sampling_rate)
    return np.squeeze(audio)

def extract_features(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate , n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs

def predict_wake_word(audio):
    features = extract_features(audio)
    features = np.expand_dims(features, axis=0)
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
        print("label: ",label)
        print("confidence: ",confidence)
        if label == 1 and confidence >0.98:
            print("✅ Wake word detected! (Confidence: {:.2f})".format(confidence))
        else:
            print("❌ No wake word. (Confidence: {:.2f})".format(confidence))
        time.sleep(0.1)  # slight pause to avoid overlapping recordings
        


# Start listener thread
listener_thread = threading.Thread(target=wake_word_listener)
listener_thread.start()

# Wait for user to stop
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    stop_flag = True
    listener_thread.join()
    print("🔚 Stopped wake word detection.")        

