# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 15:28:44 2025

@author: sudha
"""

import os
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import random

WAKE_WORD_TEXT = "hey MIKE"
NUM_WAKE_SAMPLES = 100
NUM_NON_WAKE_SAMPLES = 200
OUTPUT_DIR = "synthetic_dataset"
WAKE_WORD_AUDIO = "hey_ai_base.wav"  # Your own recorded sample
BACKGROUND_NOISE_DIR = "noises"
RANDOM_SPEECH_DIR = "random_speech"

os.makedirs(f"{OUTPUT_DIR}/wakeword", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/nonwakeword", exist_ok=True)


# Load base wake word
wake_audio = AudioSegment.from_wav(WAKE_WORD_AUDIO)

# Generate wake word samples (add noise, change speed/pitch)
for i in range(NUM_WAKE_SAMPLES):
    noisy = wake_audio + np.random.randint(-5, 5)  # volume variation
    speed = np.random.uniform(0.95, 1.05)
    altered = noisy._spawn(noisy.raw_data, overrides={
        "frame_rate": int(noisy.frame_rate * speed)
    }).set_frame_rate(noisy.frame_rate)
    altered.export(f"{OUTPUT_DIR}/wakeword/hey_ai_{i}.wav", format="wav")

# Generate non-wakeword samples
def generate_noise_sample(index):
    noise_files = os.listdir(BACKGROUND_NOISE_DIR)
    noise_file = os.path.join(BACKGROUND_NOISE_DIR, random.choice(noise_files))
    noise = AudioSegment.from_wav(noise_file)
    clip = noise[:2000]  # 2 sec
    clip.export(f"{OUTPUT_DIR}/nonwakeword/noise_{index}.wav", format="wav")

def generate_random_speech(index):
    speech_files = os.listdir(RANDOM_SPEECH_DIR)
    speech_file = os.path.join(RANDOM_SPEECH_DIR, random.choice(speech_files))
    speech = AudioSegment.from_wav(speech_file)
    clip = speech[:2000]
    clip.export(f"{OUTPUT_DIR}/nonwakeword/random_{index}.wav", format="wav")

for i in range(NUM_NON_WAKE_SAMPLES):
    if random.random() < 0.5:
        generate_noise_sample(i)
    else:
        generate_random_speech(i)
