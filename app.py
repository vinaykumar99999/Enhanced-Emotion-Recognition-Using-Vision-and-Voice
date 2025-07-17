import streamlit as st
import subprocess
import os
import sys
import soundfile as sf
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import tempfile
import signal
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time
import cv2
from realtimedetection import FaceAnalyzer
import threading
import queue
# Remove the st_audiorec import and usage

# Set page configuration
st.set_page_config(
    page_title="Enhanced Emotion Recognition",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Main background and text colors */
    .stApp {
        background-color: white;
        color: black;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        color: black;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    /* Section boxes styling */
    .section-box {
        background-color: white;
        border: 2px solid #4B8BBE;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Sub header styling */
    .sub-header {
        font-size: 1.5rem;
        color: black;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    
    /* Text styling */
    .info-text {
        font-size: 1.1rem;
        color: black;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
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
    
    /* Status text styling */
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
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Enhanced Emotion Recognition Through Vision And Voice</h1>', unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = FaceAnalyzer()
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=2)
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

def process_frame(frame):
    """Process frame in a separate thread."""
    try:
        processed_frame = st.session_state.analyzer.process_frame(frame)
        if not st.session_state.frame_queue.full():
            st.session_state.frame_queue.put(processed_frame)
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")

def main():
    st.title("Face and Voice Emotion Analysis")
    
    # Sidebar controls
    st.sidebar.title("Controls")
    start_button = st.sidebar.button("Start Analysis")
    stop_button = st.sidebar.button("Stop Analysis")
    record_button = st.sidebar.button("Start/Stop Voice Recording")
    
    # Main display area
    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    
    if start_button:
        st.session_state.is_processing = True
        cap = cv2.VideoCapture(0)
        
        while st.session_state.is_processing:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            
            # Process frame in a separate thread
            if not st.session_state.frame_queue.full():
                threading.Thread(target=process_frame, args=(frame,)).start()
            
            # Display processed frame
            if not st.session_state.frame_queue.empty():
                processed_frame = st.session_state.frame_queue.get()
                frame_placeholder.image(processed_frame, channels="BGR")
            
            # Update info
            info_placeholder.text(f"Processing: {st.session_state.is_processing}")
            
            # Handle voice recording
            if record_button:
                if not st.session_state.is_recording:
                    st.session_state.analyzer.start_voice_recording()
                    st.session_state.is_recording = True
                else:
                    emotion, _ = st.session_state.analyzer.stop_voice_recording()
                    st.session_state.is_recording = False
                    if emotion:
                        st.sidebar.success(f"Voice Emotion: {emotion}")
        
        cap.release()
    
    if stop_button:
        st.session_state.is_processing = False
        if st.session_state.is_recording:
            st.session_state.analyzer.stop_voice_recording()
            st.session_state.is_recording = False

# Function to run the vision-based emotion detection
def run_vision_detection():
    try:
        # Store the process ID in session state
        st.session_state.vision_process = subprocess.Popen([sys.executable, "realtimedetection.py"])
        st.markdown('<p class="success-text">Vision detection started!</p>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<p class="error-text">Error running vision detection: {str(e)}</p>', unsafe_allow_html=True)

# Function to stop the vision-based emotion detection
def stop_vision_detection():
    try:
        if hasattr(st.session_state, 'vision_process') and st.session_state.vision_process:
            try:
                # Get the process and its children
                parent = psutil.Process(st.session_state.vision_process.pid)
                children = parent.children(recursive=True)
                
                # Terminate all processes
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                
                try:
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass
                
                # Wait for processes to terminate
                for child in children:
                    try:
                        child.wait(timeout=3)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        pass
                
                try:
                    parent.wait(timeout=3)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
                
                # Force kill if still running
                for child in children:
                    try:
                        if child.is_running():
                            child.kill()
                    except psutil.NoSuchProcess:
                        pass
                
                try:
                    if parent.is_running():
                        parent.kill()
                except psutil.NoSuchProcess:
                    pass
                
                st.session_state.vision_process = None
                st.markdown('<p class="success-text">Vision detection stopped!</p>', unsafe_allow_html=True)
            except psutil.NoSuchProcess:
                st.markdown('<p class="warning-text">Process already terminated.</p>', unsafe_allow_html=True)
                st.session_state.vision_process = None
        else:
            st.markdown('<p class="warning-text">No vision detection process is running.</p>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<p class="error-text">Error stopping vision detection: {str(e)}</p>', unsafe_allow_html=True)
        # Reset the process state
        st.session_state.vision_process = None

# Function to record audio with progress bar
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

# Function to preprocess audio (noise reduction and normalization)
def preprocess_audio(audio_data, sample_rate):
    try:
        # Convert to mono if stereo
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Apply noise reduction (simple spectral gating)
        # Compute the spectrogram
        D = librosa.stft(audio_data)
        
        # Compute the magnitude spectrogram
        S, phase = librosa.magphase(D)
        
        # Estimate noise floor (using the bottom 10% of magnitudes)
        noise_floor = np.percentile(np.abs(S), 10, axis=1, keepdims=True)
        
        # Apply spectral gating (subtract noise floor and set negative values to 0)
        S_clean = np.maximum(np.abs(S) - noise_floor, 0) * np.exp(1j * np.angle(S))
        
        # Reconstruct the audio
        audio_clean = librosa.istft(S_clean)
        
        # Normalize again after noise reduction
        audio_clean = librosa.util.normalize(audio_clean)
        
        return audio_clean
    except Exception as e:
        print(f"Error in audio preprocessing: {str(e)}")
        return audio_data  # Return original audio if preprocessing fails

# Function to extract audio features with enhanced feature set
def extract_features(audio_data, sample_rate):
    try:
        # Preprocess audio
        audio_clean = preprocess_audio(audio_data, sample_rate)
        
        # Extract MFCCs with more coefficients
        mfccs = librosa.feature.mfcc(y=audio_clean, sr=sample_rate, n_mfcc=20)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Extract delta and delta-delta MFCCs (captures temporal changes)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Extract other features
        chroma = librosa.feature.chroma_stft(y=audio_clean, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio_clean, sr=sample_rate)
        
        # Extract zero crossing rate (useful for distinguishing emotions)
        zcr = librosa.feature.zero_crossing_rate(audio_clean)
        
        # Extract spectral centroid (brightness of sound)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_clean, sr=sample_rate)
        
        # Extract spectral rolloff (frequency below which most of the signal's energy is concentrated)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_clean, sr=sample_rate)
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=audio_clean)
        
        # Combine all features
        features = np.concatenate([
            mfccs_scaled,
            np.mean(delta_mfccs, axis=1),
            np.mean(delta2_mfccs, axis=1),
            np.mean(chroma, axis=1),
            np.mean(mel, axis=1),
            np.mean(zcr),
            np.mean(spectral_centroid),
            np.mean(spectral_rolloff),
            np.mean(rms)
        ])
        
        return features
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        # Fallback to basic feature extraction if enhanced extraction fails
        mfccs = librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=13)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        chroma = librosa.feature.chroma_stft(y=audio_data.flatten(), sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio_data.flatten(), sr=sample_rate)
        
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

# Function to predict emotion from voice with enhanced confidence calculation
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
        
        # Apply temperature scaling for better confidence calibration
        temperature = 1.5  # Adjust this value to control confidence spread
        scaled_prediction = prediction / temperature
        scaled_prediction = np.exp(scaled_prediction) / np.sum(np.exp(scaled_prediction), axis=1, keepdims=True)
        
        # Get the predicted label (always return the emotion with highest confidence)
        predicted_label = label_encoder.inverse_transform([np.argmax(scaled_prediction)])
        
        # Get confidence scores for all emotions
        confidence_scores = scaled_prediction[0]
        
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

# Initialize session state for process tracking
if 'vision_process' not in st.session_state:
    st.session_state.vision_process = None

# Create two columns for the main content
col1, col2 = st.columns(2)

# Vision section
with col1:
    # Create a container for the vision section
    vision_container = st.container()
    
    with vision_container:
        # Add the box with HTML
        st.markdown("""
        <div style="background-color: white; border: 2px solid #4B8BBE; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
            <h2 style="font-size: 1.5rem; color: black; margin-bottom: 1rem; font-weight: bold;">Vision-based Emotion Recognition</h2>
            <p style="font-size: 1.1rem; color: black; margin-bottom: 1rem;">Real-time facial expression analysis for emotion, age, and gender detection.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for start and stop buttons
        vision_col1, vision_col2 = st.columns(2)
        
        with vision_col1:
            if st.button("Start Vision Detection", key="vision_btn"):
                run_vision_detection()
        
        with vision_col2:
            if st.button("End Vision Detection", key="end_vision_btn"):
                stop_vision_detection()
                
        # Add status indicator
        if st.session_state.vision_process:
            st.markdown('<p class="success-text">Vision detection is running</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="warning-text">Vision detection is not running</p>', unsafe_allow_html=True)

# Voice section
with col2:
    # Create a container for the voice section
    voice_container = st.container()
    
    with voice_container:
        # Add the box with HTML
        st.markdown("""
        <div style="background-color: white; border: 2px solid #4B8BBE; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
            <h2 style="font-size: 1.5rem; color: black; margin-bottom: 1rem; font-weight: bold;">Voice-based Emotion Recognition</h2>
            <p style="font-size: 1.1rem; color: black; margin-bottom: 1rem;">Analyze voice patterns to detect emotions from speech input.</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Duration slider
        duration = st.slider("Recording Duration (seconds)", min_value=3, max_value=10, value=5)
        
        st.markdown('<h3 class="sub-header">Upload Your Voice (.wav file)</h3>', unsafe_allow_html=True)
        audio_file = st.file_uploader("Upload a .wav file for emotion recognition", type=["wav"])
        if audio_file is not None:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_file.read())
                temp_file.flush()
                st.audio(temp_file.name)
                # Load audio for feature extraction
                audio_data, sample_rate = sf.read(temp_file.name)
                st.markdown('<h3 class="sub-header">Analyzing Voice...</h3>', unsafe_allow_html=True)
                features = extract_features(audio_data, sample_rate)
                emotion, confidence_scores = predict_emotion(features)
                if emotion != "Model Not Trained" and emotion != "Error":
                    st.markdown('<h3 class="sub-header">Detection Result</h3>', unsafe_allow_html=True)
                    st.markdown(f'<p class="success-text">Detected Emotion: {emotion}</p>', unsafe_allow_html=True)
                    if confidence_scores is not None:
                        st.markdown('<h4 class="sub-header">Confidence Scores</h4>', unsafe_allow_html=True)
                        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
                        fig = visualize_emotions(emotions, confidence_scores)
                        st.pyplot(fig)
            os.unlink(temp_file.name)

# Footer
st.markdown('<div class="footer">Enhanced Emotion Recognition System | Powered by Deep Learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 