# Wake-word_detector
Wake-word detection system using deep learning to recognize custom trigger words in real time. This wake word projectt is used in a raspberry PI hardware so enable instant response from AI chatbot that helps with meeting recordings, query answering etc.. Since this is used in a professional environment either at universities or at work places, the accuracy of the detector to wake up when a wake-word is told heppends to be crucial. The wake word considered for this project is "HEY MIKE" which is clear, conside and an apt choice resembling a "mic".
This project went through 3 different development stages with the model's accuraacy and real time efficiency being put  to test with every milestone

# DATASET
1️) Data Collection

A custom dataset is used recorded live with voices from different people from the club that would be using this and also from a few professors. The data collected was not without a background noise, all ttypes of environments where considered and that makes the dataset and the model more robust to sensitive variations in real time. 
There is a need to specifically address the need for an imbalanced dataset as the model needs to know when not to get activated. In other words, "false positives" should be as minimum as possible. The onus shifts to the formation of non-wakeword + background noise data that truly differentiates a good model from a great one. 
Thus, we have put together 1000 samples of non-wake word and background noises along with 600 samples of wake-words. 

The dataset if then augmented to make the data a scaled up/down version by -:
1. Pitch shifting
2. noise reduction
3. Reverb

NOTE:-   Time stretching is not used as in real time the activity/hearing period is not going to be stretched.

Includes audio preprocessing, feature extraction (MFCCs), model training, and inference pipeline for integration into voice-controlled applications.

#  Features
 Audio Preprocessing: Noise reduction, silence trimming, data augmentation.
 Feature Extraction: 
       1.MFCCs (Mel-Frequency Cepstral Coefficients) for robust audio representation
       2. Delta to capture velocity and acceleartion of the audio signals. (  makes it a robust representation of audio signals )
 Model Training: 
       1. Deep learning model (feed forward NN, CNN) trained to classify wake-word vs. background speech.
       2. Stage by stage comparison/ model tuning
 Real-time Inference: Low-latency prediction pipeline for continuous audio streams.

2️) Preprocessing

Converted audio to a consistent format (mono, 16kHz).
Trimmed silence and normalized loudness.
Applied noise filtering to improve signal quality.

3️) Feature Extraction

Extracted MFCCs (Mel-Frequency Cepstral Coefficients) from each audio frame to capture spectral envelope information relevant to human perception of sound.
Computed Delta and Delta-Delta coefficients (first and second derivatives of MFCCs) to capture temporal dynamics (velocity and acceleration of spectral changes).
Concatenated MFCCs, Delta, and Delta-Delta → producing a robust feature representation with both spectral and temporal information.
Stored features as NumPy arrays for efficient handling during model training.
Saved the feature dataset in Pickle format, which preserves numerical datatypes and avoids issues where pandas.read_csv() might interpret values as strings by default.

4️) Data Augmentation

Added random noise (simulating real environments).
Time-shifted and pitch-shifted audio to increase robustness.
Applied speed variation to mimic different speaking rates.


# Model Architectures
Feed Forward Neural Network (FFNN):

Input: Flattened MFCC + Delta + Delta-Delta feature vectors per frame sequence.
Several dense layers with ReLU activation and dropout for regularization.
Output: Binary classification (Wake Word vs. Background Speech).

Convolutional Neural Network (CNN):

Input: 2D MFCC feature maps (time × coefficients).
Multiple convolutional with (3,3) filter size + max pooling layers to capture local spectral-temporal patterns.
 1----conv(3,3) followed by maxpooling and dropout
 2----conv(3,3) followed by maxpooling and dropout
 3----flattening with 128 neurons , dropout and softmax activation layer

Use of Adam optimizer and categorical cross entropy.

#  Results & Model Comparison

feed forward neural network yeilded a test accuracy of 72 % 
Accuracy: ~78%

False Positives: Higher, especially in noisy environments.

2️) Convolutional Neural Network (CNN)

Input: MFCCs treated as 2D “images” (time × frequency).
Pros: Learns local patterns in speech (spectro-temporal features).
Cons: Slightly more computationally expensive.

Accuracy: ~92%

False Positives: Much lower, robust under background noise.



# Real time Testing 
The system implements real-time wake-word detection using a pre-trained CNN model. Audio is captured from the microphone using the sounddevice library at a fixed sampling rate and duration, ensuring consistency with the training data. For each audio frame, MFCCs along with delta and delta-delta features are extracted and stacked to form a robust time–frequency representation. The feature array is then padded or truncated to a fixed length (fixed_T) and reshaped to match the CNN input requirements.

A separate listener thread continuously records audio and predicts the presence of the wake word using the trained model. Predictions include both the label and a confidence score, and wake-word detection is triggered when the confidence exceeds a defined threshold (e.g., 0.8). All features are stored in memory for later inspection, and the system prints real-time feedback to the console, indicating whether the wake word was detected. The multi-threaded design allows the listener to run continuously while the main program waits for a user interrupt (KeyboardInterrupt) to stop detection gracefully.

This setup demonstrates a low-latency, end-to-end pipeline for wake-word detection that handles feature extraction, model inference, and real-time user feedback in a production-like environment.

# code environment and applucation screenshots
<img width="959" height="429" alt="image" src="https://github.com/user-attachments/assets/53e24769-d157-401c-afc8-11d53c81ff6e" />

# Conclusion

The project demonstrate that Convolutional Neural Networks (CNNs) significantly outperform Feed-Forward Neural Networks (FFNNs) for wake-word detection. While FFNNs provide a lightweight baseline with faster training and lower resource requirements, they fail to fully exploit the structured nature of audio data. In contrast, CNNs effectively capture the time–frequency patterns present in MFCC and delta-based features, allowing them to model both local and global dependencies in the speech signal.

As a result, CNNs achieved higher accuracy, better recall, and a reduced false-positive rate, particularly in noisy or variable background conditions. This makes CNN-based architectures more suitable for real-world deployment, where robustness to diverse acoustic environments is critical. Although CNNs come with higher computational costs, their superior performance justifies their use for production-ready wake-word detection systems.
