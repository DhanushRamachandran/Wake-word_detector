import os
import librosa
import soundfile as sf
import numpy as np
import random
from tqdm import tqdm

AUGMENTATIONS_PER_FILE = 10
TARGET_SR = 44100  # or your sampling rate
DURATION = 2  # seconds
TARGET_LENGTH = TARGET_SR * DURATION

input_dir = r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\Data"
output_dir = r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\Aug_data"

os.makedirs(output_dir, exist_ok=True)

# def add_white_noise(y, noise_level=0.005):
#     return y + noise_level * np.random.randn(len(y))

def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def apply_reverb(y, reverb_level=0.3):
    reverb = np.convolve(y, np.random.rand(2000) - 0.5, mode='full')
    reverb = reverb[:len(y)]
    return y * (1 - reverb_level) + reverb * reverb_level

def random_crop(y, length):
    if len(y) <= length:
        return np.pad(y, (0, length - len(y)))
    start = random.randint(0, len(y) - length)
    return y[start:start + length]

def pad_or_trim(y, length):
    if len(y) < length:
        return np.pad(y, (0, length - len(y)))
    else:
        return y[:length]

def augment_audio(file_path, output_subdir, basename):
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    #y = pad_or_trim(y, TARGET_LENGTH)

    for i in range(AUGMENTATIONS_PER_FILE):
        aug = y.copy()

        # if i % 2 == 0:
        #     aug = add_white_noise(aug, noise_level=0.004 + 0.002 * random.random())

        if i % 2 == 0:
            aug = pitch_shift(aug, sr, n_steps=random.choice([-2, -1, 1, 2]))

        # if i % 3 == 0:
        #     rate = random.uniform(0.9, 1.1)
        #     aug = time_stretch(aug, rate)
        #     aug = pad_or_trim(aug, TARGET_LENGTH)
        
        else:
            if i % 3 == 0:
                aug = apply_reverb(aug, reverb_level=0.25)
            else:
                aug = pitch_shift(aug, sr, n_steps=random.choice([ 1, 2]))
                aug = apply_reverb(aug, reverb_level=0.25)
                

        # if i % 5 == 0:
        #     aug = random_crop(aug, TARGET_LENGTH)

        output_filename = f"{basename}_aug{i+1}.wav"
        output_path = os.path.join(output_subdir, output_filename)
        sf.write(output_path, aug, sr)

def main():
    for cls in os.listdir(input_dir):
        input_cls_dir = os.path.join(input_dir, cls)
        output_cls_dir = os.path.join(output_dir, cls)
        os.makedirs(output_cls_dir, exist_ok=True)

        for file in tqdm(os.listdir(input_cls_dir), desc=f"Augmenting class '{cls}'"):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(input_cls_dir, file)
            basename = os.path.splitext(file)[0]
            augment_audio(file_path, output_cls_dir, basename)

if __name__ == "__main__":
    main()
