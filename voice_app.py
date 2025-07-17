import streamlit as st
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import tempfile
import time
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Voice Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B8BBE;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #306998;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1.1rem;
        color: #333;
    }
    .success-text {
        font-size: 1.3rem;
        color: #28a745;
        font-weight: bold;
    }
    .warning-text {
        font-size: 1.1rem;
        color: #ffc107;
    }
    .error-text {
        font-size: 1.1rem;
        color: #dc3545;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        background-color: #4B8BBE;
        color: white;
        border-radius: 5px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #306998;
    }
    .emotion-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">Voice Emotion Recognition</h1>', unsafe_allow_html=True)
st.markdown('<p class="info-text">This application analyzes emotions from voice input using advanced machine learning techniques.</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microphone.png", width=100)
    st.markdown('<h2 class="sub-header">About</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="info-text">
    This application uses a deep learning model to detect emotions from voice recordings.
    The model has been trained on various voice samples to recognize 7 different emotions:
    </p>
    """, unsafe_allow_html=True)
    
    emotions = ['Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Disgust', 'Surprise']
    for emotion in emotions:
        st.markdown(f'<div class="emotion-card">{emotion}</div>', unsafe_allow_html=True)
    
    st.markdown('<h2 class="sub-header">Instructions</h2>', unsafe_allow_html=True)
    st.markdown("""
    <p class="info-text">
    1. Adjust the recording duration using the slider<br>
    2. Click "Record Voice" to start recording<br>
    3. Speak clearly when prompted<br>
    4. Wait for the emotion detection result
    </p>
    """, unsafe_allow_html=True)

# Function to record audio
def record_audio(duration=5, sample_rate=22050):
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Countdown message
    countdown_placeholder = st.empty()
    countdown_placeholder.markdown('<p class="info-text">Get ready...</p>', unsafe_allow_html=True)
    
    # Countdown
    for i in range(3, 0, -1):
        countdown_placeholder.markdown(f'<p class="info-text">Recording in {i}...</p>', unsafe_allow_html=True)
        time.sleep(1)
        progress_bar.progress((3-i)/3)
    
    countdown_placeholder.markdown('<p class="info-text">Speak now!</p>', unsafe_allow_html=True)
    
    # Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    
    # Update progress during recording
    for i in range(duration):
        time.sleep(1)
        progress_bar.progress((i+1)/duration)
    
    sd.wait()
    progress_bar.progress(1.0)
    
    return recording, sample_rate

# Function to extract audio features
def extract_features(audio_data, sample_rate):
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=13)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    # Extract other features
    chroma = librosa.feature.chroma_stft(y=audio_data.flatten(), sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=audio_data.flatten(), sr=sample_rate)
    
    # Combine features
    features = np.concatenate([
        mfccs_scaled,
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1)
    ])
    
    return features

# Function to check if model files exist
def check_model_files():
    model_file = 'voice_emotion_model.h5'
    encoder_file = 'voice_emotion_label_encoder.pkl'
    
    if not os.path.exists(model_file) or not os.path.exists(encoder_file):
        return False
    return True

# Function to predict emotion from voice
def predict_emotion(features):
    try:
        # Check if model files exist
        if not check_model_files():
            st.markdown('<p class="warning-text">Voice emotion model not found. Please follow these steps:</p>', unsafe_allow_html=True)
            st.markdown("""
            <ol class="info-text">
                <li>Add voice samples to the voice_data directory</li>
                <li>Run 'python train_voice_model.py' to train the model</li>
                <li>Restart this application</li>
            </ol>
            """, unsafe_allow_html=True)
            return "Model Not Trained", None
        
        # Load the model and label encoder
        model = load_model('voice_emotion_model.h5')
        label_encoder = joblib.load('voice_emotion_label_encoder.pkl')
        
        # Reshape features for prediction
        features = features.reshape(1, features.shape[0], 1)
        
        # Make prediction
        prediction = model.predict(features)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        
        # Get confidence scores for all emotions
        confidence_scores = prediction[0]
        
        return predicted_label[0], confidence_scores
    except Exception as e:
        st.markdown(f'<p class="error-text">Error loading model or making prediction: {str(e)}</p>', unsafe_allow_html=True)
        return "Error", None

# Function to visualize emotion confidence
def visualize_emotions(emotions, confidence_scores):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create a bar chart
    bars = ax.bar(emotions, confidence_scores, color='skyblue')
    
    # Add labels and title
    ax.set_xlabel('Emotions', fontsize=12)
    ax.set_ylabel('Confidence', fontsize=12)
    ax.set_title('Emotion Confidence Scores', fontsize=14)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Set y-axis limit
    ax.set_ylim(0, 1.0)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Main content
st.markdown('<h2 class="sub-header">Voice-based Emotion Recognition</h2>', unsafe_allow_html=True)
st.markdown('<p class="info-text">Click the button below to record your voice for emotion detection</p>', unsafe_allow_html=True)

# Check if model is trained
if not check_model_files():
    st.markdown('<p class="warning-text">Voice emotion model not trained yet. Please:</p>', unsafe_allow_html=True)
    st.markdown("""
    <ol class="info-text">
        <li>Add voice samples to the voice_data directory</li>
        <li>Run 'python train_voice_model.py' to train the model</li>
        <li>Restart this application</li>
    </ol>
    """, unsafe_allow_html=True)

# Create two columns for the slider and button
col1, col2 = st.columns([3, 1])

with col1:
    # Duration slider
    duration = st.slider("Recording Duration (seconds)", min_value=3, max_value=10, value=5)

with col2:
    # Record button
    record_button = st.button("Record Voice", key="voice_btn")

if record_button:
    try:
        # Record audio
        audio_data, sample_rate = record_audio(duration=duration)
        
        # Save the recording temporarily
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate)
            
            # Display audio player
            st.markdown('<h3 class="sub-header">Your Recording</h3>', unsafe_allow_html=True)
            st.audio(temp_file.name)
        
        # Extract features
        st.markdown('<h3 class="sub-header">Analyzing Voice...</h3>', unsafe_allow_html=True)
        features = extract_features(audio_data, sample_rate)
        
        # Predict emotion
        emotion, confidence_scores = predict_emotion(features)
        
        # Display result
        if emotion != "Model Not Trained" and emotion != "Error":
            st.markdown('<h3 class="sub-header">Detection Result</h3>', unsafe_allow_html=True)
            
            # Display the detected emotion
            st.markdown(f'<p class="success-text">Detected Emotion: {emotion}</p>', unsafe_allow_html=True)
            
            # Visualize confidence scores
            if confidence_scores is not None:
                st.markdown('<h4 class="sub-header">Confidence Scores</h4>', unsafe_allow_html=True)
                fig = visualize_emotions(emotions, confidence_scores)
                st.pyplot(fig)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
    except Exception as e:
        st.markdown(f'<p class="error-text">Error during voice recording or processing: {str(e)}</p>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Voice Emotion Recognition System | Powered by Machine Learning</div>', unsafe_allow_html=True) 