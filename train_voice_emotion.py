import os
import numpy as np
import librosa
import sounddevice as sd
import threading
import time
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

class VoiceEmotionTrainer:
    def __init__(self):
        self.sample_rate = 44100
        self.recording_duration = 3  # seconds
        self.is_recording = False
        self.audio_data = []
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                             4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        self.recordings = []
        self.labels = []
        self.model = None
        
    def create_model(self):
        """Create the voice emotion recognition model."""
        inputs = Input(shape=(None, 40))
        x = LSTM(512, return_sequences=True)(inputs)
        x = Dropout(0.4)(x)
        x = LSTM(256, return_sequences=True)(x)
        x = Dropout(0.4)(x)
        x = LSTM(128)(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(7, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model
    
    def start_recording(self):
        """Start recording voice sample."""
        self.is_recording = True
        self.audio_data = []
        
        def record_audio():
            with sd.InputStream(samplerate=self.sample_rate, channels=1) as stream:
                while self.is_recording:
                    audio_chunk, _ = stream.read(1024)
                    self.audio_data.extend(audio_chunk.flatten())
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording and process the audio."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.audio_data:
            return np.array(self.audio_data)
        return None
    
    def extract_features(self, audio_data):
        """Extract features from audio data."""
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=40)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        mel = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
        
        features = np.concatenate([
            mfccs,
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate,
            spectral_bandwidth,
            spectral_contrast,
            chroma,
            mel
        ])
        
        return features
    
    def record_sample(self, emotion):
        """Record a sample for a specific emotion."""
        print(f"\nRecording sample for emotion: {emotion}")
        print("Press Enter to start recording...")
        input()
        
        self.start_recording()
        print("Recording... Press Enter to stop.")
        input()
        
        audio_data = self.stop_recording()
        if audio_data is not None:
            features = self.extract_features(audio_data)
            self.recordings.append(features)
            self.labels.append(emotion)
            print(f"Sample recorded for {emotion}")
            return True
        return False
    
    def save_data(self, filename='voice_emotion_data.pkl'):
        """Save recorded data to file."""
        data = {
            'recordings': self.recordings,
            'labels': self.labels
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filename}")
    
    def load_data(self, filename='voice_emotion_data.pkl'):
        """Load recorded data from file."""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.recordings = data['recordings']
                self.labels = data['labels']
            print(f"Data loaded from {filename}")
            return True
        return False
    
    def train_model(self, epochs=50, batch_size=32):
        """Train the model on recorded data."""
        if not self.recordings or not self.labels:
            print("No data available for training")
            return
        
        # Convert labels to one-hot encoding
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(self.labels)
        labels_onehot = np.eye(len(self.emotion_labels))[labels_encoded]
        
        # Prepare data
        X = np.array(self.recordings)
        y = labels_onehot
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = self.create_model()
        
        print("\nTraining model...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model
        model.save('voice_emotion_model.h5')
        print("Model saved as voice_emotion_model.h5")
        
        return history

def main():
    trainer = VoiceEmotionTrainer()
    
    # Try to load existing data
    if trainer.load_data():
        print(f"Loaded {len(trainer.recordings)} samples")
    
    while True:
        print("\nVoice Emotion Training Menu:")
        print("1. Record new samples")
        print("2. Train model")
        print("3. Save data")
        print("4. Load data")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            print("\nAvailable emotions:")
            for idx, emotion in trainer.emotion_labels.items():
                print(f"{idx}: {emotion}")
            
            emotion_idx = int(input("Enter emotion index (0-6): "))
            if 0 <= emotion_idx <= 6:
                trainer.record_sample(trainer.emotion_labels[emotion_idx])
            else:
                print("Invalid emotion index")
        
        elif choice == '2':
            epochs = int(input("Enter number of epochs (default 50): ") or "50")
            batch_size = int(input("Enter batch size (default 32): ") or "32")
            trainer.train_model(epochs=epochs, batch_size=batch_size)
        
        elif choice == '3':
            trainer.save_data()
        
        elif choice == '4':
            trainer.load_data()
        
        elif choice == '5':
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main() 