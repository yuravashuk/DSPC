import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Directory where your .wav files are organized in subfolders
main_directory = './data/training/'

# List for features and labels
features = []
labels = []

# Iterate over subfolders in main directory
for subfolder in os.listdir(main_directory):
    subfolder_path = os.path.join(main_directory, subfolder)
    # Only process subfolders (skip any file that might be in the main directory)
    if os.path.isdir(subfolder_path):
        # Iterate over files in subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".wav"):
                # Load .wav file
                y, sr = librosa.load(os.path.join(subfolder_path, filename))

                # Extract MEL spectrogram
                mel = librosa.feature.melspectrogram(y=y, sr=sr)

                # Extract MFCC (mel-frequency cepstral coefficients)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)

                # Append features to the feature list
                features.append(np.hstack((np.mean(mel), np.mean(mfcc))))

                # Append the label (subfolder name) to the labels list
                labels.append(subfolder)
                print(subfolder)

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Split dataset into training and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.4)

# Train model
model = RandomForestClassifier()
model.fit(features_train, labels_train)

# Save model and encoder
joblib.dump(model, '/data/models/model.joblib')
joblib.dump(encoder, '/data/models/encoder.joblib')
