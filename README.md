# Wake-word_detector
Wake-word detection system using deep learning to recognize custom trigger words in real time. Includes audio preprocessing, feature extraction (MFCCs), model training, and inference pipeline for integration into voice-controlled applications.

Features
 Audio Preprocessing: Noise reduction, silence trimming, data augmentation.
 Feature Extraction: MFCCs (Mel-Frequency Cepstral Coefficients) for robust audio representation.
 Model Training: Deep learning model (LSTM/CNN) trained to classify wake-word vs. background speech.
 Real-time Inference: Low-latency prediction pipeline for continuous audio streams.

📊 Data Pipeline
1️) Data Collection

Recorded custom audio samples for the wake word ("HEY MIKE").
Collected negative examples (background noise, speech without the wake word) from open datasets and real-world recordings.
Dataset is distributed unevenly as the onus is to not to get false positives in the first place. 
Thus 65%- noisy data,35% wake word data.

2️) Preprocessing

Converted audio to a consistent format (mono, 16kHz).
Trimmed silence and normalized loudness.
Applied noise filtering to improve signal quality.

3️) Feature Extraction

Extracted MFCCs (Mel-Frequency Cepstral Coefficients) from each audio frame.
Captured spectral patterns relevant to human speech.
Stored features as numpy arrays for efficient training.
Creating a new df and storing on the "pickle" format.
Pickle format is the best for reading data as numbers otherwise pandas read_csv takes it as astring by default.

4️) Data Augmentation

Added random noise (simulating real environments).
Time-shifted and pitch-shifted audio to increase robustness.
Applied speed variation to mimic different speaking rates.

5️) Model Building

Built a CNN/ feed forward based classifier to distinguish wake word vs. background speech.
Optimized using cross-entropy loss and Adam optimizer.
Evaluated with accuracy, recall, and false-positive rate.

Results & Model Comparison

I experimented with two different approaches for wake-word detection:

1️) Feed-Forward Neural Network (FFNN)

Input: Flattened MFCC feature vectors.

Pros: Simpler, faster to train.

Cons: Limited ability to capture temporal/spectral patterns.

Accuracy: ~78%

False Positives: Higher, especially in noisy environments.

2️) Convolutional Neural Network (CNN)

Input: MFCCs treated as 2D “images” (time × frequency).

Pros: Learns local patterns in speech (spectro-temporal features).

Cons: Slightly more computationally expensive.

Accuracy: ~92%

False Positives: Much lower, robust under background noise.

Takeaway: CNNs significantly outperform feed-forward networks for wake-word detection, as they better capture the time–frequency structure of audio signals.

