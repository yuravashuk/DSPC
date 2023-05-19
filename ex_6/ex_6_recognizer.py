import numpy as np
import librosa
import joblib

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    features = np.hstack((np.mean(mel), np.mean(mfcc)))
    return features

def classify_audio(file_path, model_path='model.joblib', encoder_path='encoder.joblib'):
    # Load the model and encoder
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # Extract features from the audio file
    features = extract_features(file_path)

    # Make a prediction
    prediction = model.predict([features])

    # Decode the prediction
    label = encoder.inverse_transform(prediction)

    return label

# Usage:
file_path = './data/testing/notepad_test.wav'
model_path = './data/models/model.joblib'
encoder_path = './data/models/encoder.joblib'

print(classify_audio(file_path=file_path, model_path=model_path, encoder_path=encoder_path))