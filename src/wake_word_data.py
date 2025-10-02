# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 19:52:49 2025

@author: sudha
"""

import sounddevice as sd
from scipy.io.wavfile import write
from wake_word_config import all_configs
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class DatasetCreator:
    
    def __init__(self):
        self.n = all_configs["no_of_dataset_files"]
        #self.n = 20
        self.sampling_rate = all_configs["sampling_configs"]["sampling_rate"]
        self.sec = all_configs["sampling_configs"]["time"]
        self.sec =2
        self.dataset_dir = all_configs["dataset_dir"]

    def record_audio(self, filename_prefix):
        for file_count in range(1, self.n + 1):
            input(f"[{file_count}/{self.n}] Press Enter to record ({filename_prefix})...")
            print("Recording... Speak now!")
            
            my_recording = sd.rec(
                int(self.sec * self.sampling_rate),
                samplerate=self.sampling_rate,
                channels=1,
                dtype='int16'
            )
            sd.wait()

            filename = f"{filename_prefix}_{file_count}.wav"
            save_path = os.path.join(self.dataset_dir, filename)
            write(save_path, self.sampling_rate, my_recording)

            print(f"Audio file saved at: {save_path}\n")

    def record_and_save_wake_word(self):
        self.record_audio("wake_dataset")

    def record_and_save_backgrnd_noise(self):
        self.record_audio("wake_bgn_dataset")
        
    def view_data_sample(self):
        for file in os.listdir(all_configs["bgn_dir"]):
            #if "bgn" not in file:
                data,sample_rate = librosa.load(os.path.join(all_configs["bgn_dir"],file))
                plt.figure()
                plt.title(f"Waveform of {file}")
                print(sample_rate)
                print(data)
                librosa.display.waveshow(data,sr=sample_rate)
                plt.show()
                
               
# data preprocessing

def gen_data_preprocessing():
    
    wake_words_dir = all_configs["wake_words_dir"]
    bgn_dir = all_configs["bgn_dir"]
    
    
    data_path_dict = {
        1:  [os.path.join(wake_words_dir,file) for file in os.listdir(wake_words_dir)],
        0:  [os.path.join(bgn_dir,file) for file in os.listdir(bgn_dir)] 
            }
    
    n_mfcc = all_configs["feature_extraction"]["n_mfcc"]
    mfcc_features = []
    for class_label,list_of_files in data_path_dict.items():
        for audio_file in list_of_files:
            data,sample_rate = librosa.load(audio_file)
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc)  
    
            # Optional: Take mean over time axis to get a fixed-size feature vector
            mfcc_mean = np.mean(mfcc.T, axis=0)  
    
            mfcc_features.append([mfcc_mean, class_label])  # Save as tuple (features, label)
            
    
    df = pd.DataFrame(mfcc_features,columns=["feature","class"])
    df.to_pickle(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\Data/full_dataset_4.csv")
    
    
def delta_based_preprocessing():
        # Directories
    wake_words_dir = all_configs["wake_words_dir"]
    bgn_dir = all_configs["bgn_dir"]
    
    # Data path mapping: class label -> list of file paths
    data_path_dict = {
        1: [os.path.join(wake_words_dir, file) for file in os.listdir(wake_words_dir)],
        0: [os.path.join(bgn_dir, file) for file in os.listdir(bgn_dir)] 
    }
    
    # Feature extraction configs
    n_mfcc = all_configs["feature_extraction"]["n_mfcc"]
    mfcc_features = []
    
    for class_label, list_of_files in data_path_dict.items():
        for audio_file in list_of_files:
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
    
    
    # Save to DataFrame
    df = pd.DataFrame(mfcc_features, columns=["feature", "class"])
    print("df len: ",len(df))
    # Save to disk
    df.to_pickle(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\Data/sample_tesing.csv")
        


# Create object and record wake word samples
if __name__ == "__main__":
    
    #dataset_creator = DatasetCreator()
    #dataset_creator.record_and_save_wake_word()
    #dataset_creator.record_and_save_backgrnd_noise()  # Uncomment to run background recording too
    #dataset_creator.view_data_sample()
    #gen_data_preprocessing()
    delta_based_preprocessing()
    