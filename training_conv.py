# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 23:29:27 2025

@author: sudha
"""

import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from wake_word_config import all_configs

tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)

# Configuration
n_mfcc = all_configs["feature_extraction"]["n_mfcc"]
n_classes = 2

# Load dataset
df = pd.read_pickle(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\Data\full_dataset_delta_1_800_samp.csv")
#df = df[df["class"].isin([0])]  # Optional filtering

# Convert to arrays
x = df["feature"].values
y = df["class"].values

# Rebuild the 2D array from flattened (39*T,)
# Assuming each was originally (39, T) before .mean()
# If you saved it flattened, reshape appropriately
x = np.array([feature for feature in x])  # shape: (samples, 39, T)
if len(x.shape) == 2:
    raise ValueError(f"CNN needs 2D features ({n_mfcc}, T), not meaned vectors. Go back and remove mean().")

# Add channel dimension for CNN
x = x[..., np.newaxis]  # shape: (samples, 39, T, 1)

# Encode labels
y = to_categorical(y, num_classes=n_classes)

# Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)

                                                    
# CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=x.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, epochs=7, batch_size=32, validation_split=0.1)


# Evaluate
score = model.evaluate(x_test, y_test)
print("Test Score:", score)

# Predict
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Classification Report:\n", classification_report(y_true, y_pred))


# Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["non-wake", "wake"], yticklabels=["non-wake", "wake"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Save model
model.save(r"C:\Users\sudha\Desktop\dhanush\Personal DS\NLP\wake_word project\model\Hey_mike_CNN_model32_800_samp.h5")
