import cv2
import numpy as np
from keras.models import model_from_json, load_model
import traceback
import time
import os
import librosa
# import sounddevice as sd  # Removed for Streamlit Cloud compatibility
import threading
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D

class FaceAnalyzer:
    def __init__(self):
        self.age_net = None
        self.gender_net = None 
        self.emotion_model = None
        self.voice_emotion_model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Enhanced age ranges for better granularity
        self.age_list = ['0-2', '4-6', '8-12', '15-20', '21-25', '26-32', '33-40', '41-50', '51-60', '60+']
        self.gender_list = ['Male', 'Female']
        self.emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 
                               4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
        
        # Improved thresholds
        self.emotion_threshold = 0.6  # Increased for more confident predictions
        self.age_threshold = 0.3     # Balanced threshold
        self.child_threshold = 0.5    # Increased for better child detection
        
        # Enhanced temporal smoothing
        self.age_history = []
        self.gender_history = []
        self.emotion_history = []
        self.voice_emotion_history = []
        self.history_size = 10  # Increased for better smoothing
        
        # Improved confidence thresholds
        self.age_confidence_threshold = 0.3
        self.gender_confidence_threshold = 0.6
        self.emotion_confidence_threshold = 0.6
        self.voice_emotion_confidence_threshold = 0.6
        
        # Previous predictions for stability
        self.prev_age = None
        self.prev_gender = None
        self.prev_emotion = None
        self.prev_voice_emotion = None
        
        # Gender bias correction
        self.gender_bias_correction = 0.15  # Increased for better female detection
        
        # Voice recording parameters
        self.sample_rate = 44100
        self.recording_duration = 3  # seconds
        self.is_recording = False
        self.audio_data = None
        
        # Initialize ResNet50 model for gender detection
        self.resnet_gender_model = None
        
        self.load_models()
        self.initialize_voice_emotion_model()
        self.initialize_resnet_gender_model()
    
    def initialize_resnet_gender_model(self):
        """Initialize the ResNet50 model for gender detection with improved architecture."""
        try:
            # Load pre-trained ResNet50 model
            base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            
            # Add custom layers for gender detection
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(2048, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(2, activation='softmax')(x)
            
            # Create the model
            self.resnet_gender_model = Model(inputs=base_model.input, outputs=predictions)
            
            # Compile the model
            self.resnet_gender_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load pre-trained weights if available
            if os.path.exists('resnet_gender_model.h5'):
                try:
                    self.resnet_gender_model.load_weights('resnet_gender_model.h5')
                    print("Successfully loaded ResNet gender model weights")
                except Exception as e:
                    print(f"Error loading ResNet weights: {str(e)}")
                    print("Using default weights")
            else:
                print("Warning: ResNet gender model weights not found. Using default weights.")
                # Initialize with default weights
                self.resnet_gender_model.build((None, 224, 224, 3))
            
            # Freeze the base model layers
            for layer in base_model.layers:
                layer.trainable = False
                
        except Exception as e:
            print(f"Error initializing ResNet gender model: {str(e)}")
            # Create a simple fallback model
            self.resnet_gender_model = Sequential([
                Dense(2, activation='softmax', input_shape=(224, 224, 3))
            ])
            self.resnet_gender_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def detect_gender_resnet(self, face_img):
        """Detect gender using ResNet50 model with enhanced preprocessing and validation."""
        try:
            # Enhanced preprocessing for ResNet50
            face_img = cv2.resize(face_img, (224, 224))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Apply additional preprocessing
            face_img = cv2.GaussianBlur(face_img, (3,3), 0)
            face_img = cv2.addWeighted(face_img, 1.5, cv2.GaussianBlur(face_img, (0,0), 10), -0.5, 0)
            
            # Normalize
            face_img = face_img.astype(np.float32) / 255.0
            face_img = np.expand_dims(face_img, axis=0)
            
            # Get predictions
            predictions = self.resnet_gender_model.predict(face_img, verbose=0)
            gender_idx = np.argmax(predictions[0])
            confidence = predictions[0][gender_idx]
            
            # Print detailed confidence scores
            print("\nResNet Gender Detection Details:")
            print(f"Male confidence: {predictions[0][0]:.3f}")
            print(f"Female confidence: {predictions[0][1]:.3f}")
            print(f"Selected gender: {self.gender_list[gender_idx]}")
            print(f"Confidence: {confidence:.3f}")
            
            return self.gender_list[gender_idx], confidence
        except Exception as e:
            print(f"Error in ResNet gender detection: {str(e)}")
            return None, 0.0
    
    def initialize_voice_emotion_model(self):
        """Initialize the voice emotion recognition model with improved architecture."""
        try:
            # Create model with proper input layer
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
            
            self.voice_emotion_model = Model(inputs=inputs, outputs=outputs)
            
            self.voice_emotion_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Load pre-trained weights if available
            if os.path.exists('voice_emotion_model.h5'):
                try:
                    self.voice_emotion_model.load_weights('voice_emotion_model.h5')
                    print("Successfully loaded voice emotion model weights")
                except Exception as e:
                    print(f"Error loading voice emotion weights: {str(e)}")
                    print("Using default weights")
            else:
                print("Warning: Voice emotion model weights not found. Using default weights.")
                # Initialize with default weights
                self.voice_emotion_model.build((None, None, 40))
        except Exception as e:
            print(f"Error initializing voice emotion model: {str(e)}")
            # Create a simple fallback model
            self.voice_emotion_model = Sequential([
                Dense(7, activation='softmax', input_shape=(40,))
            ])
            self.voice_emotion_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    
    def start_voice_recording(self):
        """Start recording voice for emotion analysis."""
        self.is_recording = True
        self.audio_data = []
        
        def record_audio():
            # This function is no longer directly usable for audio recording in Streamlit Cloud
            # as sounddevice is removed.
            # For Streamlit Cloud, audio input would typically come from a Streamlit component
            # or a pre-recorded audio file.
            # For now, we'll just simulate recording or raise an error.
            print("Voice recording is not available in this environment.")
            self.is_recording = False # Ensure recording stops if not available
        
        self.recording_thread = threading.Thread(target=record_audio)
        self.recording_thread.start()
    
    def stop_voice_recording(self):
        """Stop voice recording and process the audio with enhanced feature extraction and emotion validation."""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        
        if self.audio_data:
            # Convert to numpy array
            audio_data = np.array(self.audio_data)
            
            # Extract enhanced features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=40)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            mel = librosa.feature.melspectrogram(y=audio_data, sr=self.sample_rate)
            
            # Calculate additional features for emotion validation
            pitch = librosa.yin(audio_data, fmin=50, fmax=500)
            energy = librosa.feature.rms(y=audio_data)
            tempo = librosa.beat.tempo(y=audio_data, sr=self.sample_rate)
            
            # Combine features
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
            
            # Normalize features
            features = (features - np.mean(features)) / np.std(features)
            features = np.expand_dims(features, axis=0)
            
            # Predict emotion
            predictions = self.voice_emotion_model.predict(features, verbose=0)
            
            # Apply emotion-specific validation rules
            emotion_scores = predictions[0].copy()
            
            # Validate based on audio features
            if np.mean(energy) < 0.1:  # Low energy
                emotion_scores[3] *= 0.5  # Reduce happy score
                emotion_scores[4] *= 1.5  # Increase neutral score
                emotion_scores[5] *= 1.2  # Increase sad score
            
            if np.mean(pitch) > 300:  # High pitch
                emotion_scores[2] *= 1.3  # Increase fear score
                emotion_scores[6] *= 1.2  # Increase surprise score
            
            if np.mean(tempo) > 120:  # Fast tempo
                emotion_scores[0] *= 1.2  # Increase angry score
                emotion_scores[3] *= 1.2  # Increase happy score
            
            if np.mean(spectral_centroid) > 3000:  # High spectral centroid
                emotion_scores[1] *= 1.2  # Increase disgust score
                emotion_scores[2] *= 1.2  # Increase fear score
            
            # Apply emotion-specific confidence thresholds
            emotion_thresholds = {
                0: 0.5,  # Angry - higher threshold
                1: 0.5,  # Disgust - higher threshold
                2: 0.5,  # Fear - higher threshold
                3: 0.6,  # Happy - much higher threshold
                4: 0.4,  # Neutral - lower threshold
                5: 0.5,  # Sad - higher threshold
                6: 0.5   # Surprise - higher threshold
            }
            
            # Apply temporal smoothing
            self.voice_emotion_history.append((np.argmax(emotion_scores), np.max(emotion_scores)))
            if len(self.voice_emotion_history) > self.history_size:
                self.voice_emotion_history.pop(0)
            
            # Get stable prediction with enhanced validation
            if len(self.voice_emotion_history) == self.history_size:
                emotion_votes = {}
                for idx, conf in self.voice_emotion_history:
                    if conf > emotion_thresholds[idx]:
                        emotion_votes[idx] = emotion_votes.get(idx, 0) + conf
                
                if emotion_votes:
                    stable_idx = max(emotion_votes.items(), key=lambda x: x[1])[0]
                    emotion = self.emotion_labels[stable_idx]
                else:
                    # If no confident votes, use weighted average with validation
                    emotion_scores = np.zeros(7)
                    for idx, conf in self.voice_emotion_history:
                        emotion_scores[idx] += conf
                    
                    # Apply additional validation rules
                    if np.mean(energy) < 0.1:  # Low energy
                        emotion_scores[4] *= 1.5  # Boost neutral
                        emotion_scores[5] *= 1.2  # Boost sad
                    
                    if np.mean(pitch) > 300:  # High pitch
                        emotion_scores[2] *= 1.3  # Boost fear
                        emotion_scores[6] *= 1.2  # Boost surprise
                    
                    emotion = self.emotion_labels[np.argmax(emotion_scores)]
            else:
                if np.max(emotion_scores) > emotion_thresholds[np.argmax(emotion_scores)]:
                    emotion = self.emotion_labels[np.argmax(emotion_scores)]
                else:
                    emotion = self.prev_voice_emotion if self.prev_voice_emotion else "Neutral"
            
            self.prev_voice_emotion = emotion
            
            # Print detailed voice emotion detection information
            print("\nVoice Emotion Detection Details:")
            print("Raw predictions:")
            for idx, label in self.emotion_labels.items():
                print(f"{label}: {predictions[0][idx]:.3f}")
            print("\nValidated predictions:")
            for idx, label in self.emotion_labels.items():
                print(f"{label}: {emotion_scores[idx]:.3f}")
            print(f"\nSelected emotion: {emotion}")
            print(f"Confidence: {np.max(emotion_scores):.3f}")
            print("\nAudio Features:")
            print(f"Energy: {np.mean(energy):.3f}")
            print(f"Pitch: {np.mean(pitch):.3f}")
            print(f"Tempo: {np.mean(tempo):.3f}")
            print(f"Spectral Centroid: {np.mean(spectral_centroid):.3f}")
            
            return emotion, emotion_scores
        
        return None, None
    
    def load_models(self):
        """Load all models with error handling."""
        try:
            required_files = {
                'facialemotionmodel.json': 'Emotion model JSON',
                'facialemotionmodel.h5': 'Emotion model weights',
                'age_net.caffemodel': 'Age detection model',
                'age_deploy.prototxt': 'Age model configuration',
                'gender_net.caffemodel': 'Gender detection model',
                'gender_deploy.prototxt': 'Gender model configuration'
            }
            for file, desc in required_files.items():
                if not os.path.exists(file):
                    raise FileNotFoundError(f"{desc} not found: {file}")
            
            # Load emotion detection model
            with open("facialemotionmodel.json", "r") as json_file:
                model_json = json_file.read()
            try:
                self.emotion_model = model_from_json(model_json)
            except TypeError:
                self.emotion_model = load_model("facialemotionmodel.h5")
            self.emotion_model.load_weights("facialemotionmodel.h5")
            
            # Load OpenCV models
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.age_net = cv2.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
            self.gender_net = cv2.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')
            print(" All models loaded successfully!")
        except Exception as e:
            print(f" Error loading models: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def preprocess_face(self, face_img):
        """Enhanced preprocessing for face images with focus on emotion detection."""
        try:
            # Convert to LAB color space for better lighting normalization
            lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE with adjusted parameters for better feature extraction
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            lab = cv2.merge((l,a,b))
            face_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            # Apply bilateral filter with adjusted parameters for better feature preservation
            face_img = cv2.bilateralFilter(face_img, 7, 35, 35)
            
            # Apply sharpening filter for better detail
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            face_img = cv2.filter2D(face_img, -1, kernel)
            
            # Normalize image
            face_img = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply additional contrast enhancement
            face_img = cv2.convertScaleAbs(face_img, alpha=1.2, beta=10)
            
            # Apply edge enhancement for better feature detection
            face_img = cv2.addWeighted(face_img, 1.5, cv2.GaussianBlur(face_img, (0,0), 10), -0.5, 0)
            
            return face_img
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return face_img
    
    def is_child(self, face_img):
        """Enhanced child detection with more features."""
        try:
            # Calculate face aspect ratio
            height, width = face_img.shape[:2]
            aspect_ratio = width / height
            
            # Children typically have larger eyes and rounder faces
            # So we look for higher aspect ratios
            is_child_by_ratio = aspect_ratio > 0.9
            
            # Additional check: children often have smoother skin
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_child_by_texture = blur < 500  # Lower variance indicates smoother skin
            
            # Combine both checks
            return is_child_by_ratio or is_child_by_texture
        except Exception as e:
            print(f"Error in child detection: {str(e)}")
            return False
    
    def detect_face(self, frame):
        """Enhanced face detection with better parameters."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Adjusted parameters for better face detection
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,  # More gradual scaling
                minNeighbors=4,   # Reduced for better detection
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return []
    
    def detect_age_gender(self, face_img):
        """Enhanced age and gender detection with improved accuracy."""
        try:
            # Preprocess face image
            face_img = self.preprocess_face(face_img)
            face_img_resized = cv2.resize(face_img, (227, 227))
            face_img_resized = cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB)
            
            # Get gender prediction from ResNet50
            resnet_gender, resnet_conf = self.detect_gender_resnet(face_img)
            
            # Create blob for OpenCV model
            blob = cv2.dnn.blobFromImage(
                face_img_resized,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=True
            )
            
            # Get gender prediction from OpenCV model
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            
            # Print OpenCV model predictions
            print("\nOpenCV Gender Detection Details:")
            print(f"Male confidence: {gender_preds[0][0]:.3f}")
            print(f"Female confidence: {gender_preds[0][1]:.3f}")
            
            # Combine predictions from both models
            if resnet_gender and resnet_conf > 0.6:  # If ResNet is confident
                if resnet_gender == 'Female':
                    gender_preds[0][1] += 0.3  # Increase female probability
                    print("Applied female boost from ResNet")
                else:
                    gender_preds[0][0] += 0.3  # Increase male probability
                    print("Applied male boost from ResNet")
            
            # Extract facial features for additional gender cues
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Detect facial landmarks
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = face_img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Calculate facial proportions
                face_height = h
                face_width = w
                face_ratio = face_width / face_height
                
                # Calculate eye spacing and features
                if len(eyes) >= 2:
                    eye_spacing = abs(eyes[0][0] - eyes[1][0])
                    eye_ratio = eye_spacing / face_width
                    
                    # Calculate eye shape features
                    eye_areas = []
                    for (ex, ey, ew, eh) in eyes:
                        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                        eye_areas.append(ew * eh)
                    
                    eye_area_ratio = min(eye_areas) / max(eye_areas)
                else:
                    eye_ratio = 0.5
                    eye_area_ratio = 1.0
                
                # Calculate jawline and chin features
                jawline_points = []
                for (ex, ey, ew, eh) in eyes:
                    jawline_points.append((ex + ew//2, ey + eh))
                
                if len(jawline_points) >= 2:
                    jawline_angle = np.arctan2(jawline_points[1][1] - jawline_points[0][1],
                                             jawline_points[1][0] - jawline_points[0][0])
                    chin_width = abs(jawline_points[1][0] - jawline_points[0][0])
                    chin_ratio = chin_width / face_width
                else:
                    jawline_angle = 0
                    chin_ratio = 0.5
                
                # Calculate facial symmetry
                left_half = roi_gray[:, :w//2]
                right_half = cv2.flip(roi_gray[:, w//2:], 1)
                symmetry_score = np.mean(np.abs(left_half - right_half))
                
                # Calculate skin texture and smoothness
                skin_texture = cv2.Laplacian(roi_gray, cv2.CV_64F).var()
                skin_smoothness = 1.0 / (1.0 + skin_texture)
                
                # Calculate eyebrow features
                eyebrow_region = roi_gray[max(0, y-20):y, x:x+w]
                eyebrow_texture = cv2.Laplacian(eyebrow_region, cv2.CV_64F).var()
                
                # Calculate cheekbone features
                cheekbone_region = roi_gray[y:y+h//2, x:x+w]
                cheekbone_texture = cv2.Laplacian(cheekbone_region, cv2.CV_64F).var()
                
                # Calculate forehead region features
                forehead_region = roi_gray[max(0, y-h//4):y, x:x+w]
                forehead_texture = cv2.Laplacian(forehead_region, cv2.CV_64F).var()
                
                # Calculate hairline features
                hairline_region = roi_gray[max(0, y-h//3):y, x:x+w]
                hairline_edges = cv2.Canny(hairline_region, 50, 150)
                hairline_straightness = np.sum(hairline_edges) / (w * h//3)
                
                # Calculate lip features
                lip_region = roi_gray[y+h//2:y+h, x:x+w]
                lip_texture = cv2.Laplacian(lip_region, cv2.CV_64F).var()
                
                # Print feature values
                print("\nFacial Feature Analysis:")
                print(f"Face ratio: {face_ratio:.3f}")
                print(f"Eye ratio: {eye_ratio:.3f}")
                print(f"Eye area ratio: {eye_area_ratio:.3f}")
                print(f"Jawline angle: {abs(jawline_angle):.3f}")
                print(f"Chin ratio: {chin_ratio:.3f}")
                print(f"Symmetry score: {symmetry_score:.3f}")
                print(f"Skin smoothness: {skin_smoothness:.3f}")
                print(f"Eyebrow texture: {eyebrow_texture:.3f}")
                print(f"Cheekbone texture: {cheekbone_texture:.3f}")
                print(f"Forehead texture: {forehead_texture:.3f}")
                print(f"Hairline straightness: {hairline_straightness:.3f}")
                print(f"Lip texture: {lip_texture:.3f}")
                
                # Combine all features for gender prediction
                gender_features = np.array([
                    face_ratio,
                    eye_ratio,
                    eye_area_ratio,
                    abs(jawline_angle),
                    chin_ratio,
                    symmetry_score,
                    skin_smoothness,
                    eyebrow_texture,
                    cheekbone_texture,
                    forehead_texture,
                    hairline_straightness,
                    lip_texture
                ])
                
                # Normalize features
                gender_features = (gender_features - np.mean(gender_features)) / np.std(gender_features)
                
                # Weight the features (adjusted for better female detection)
                feature_weights = np.array([
                    0.07,  # face_ratio
                    0.07,  # eye_ratio
                    0.07,  # eye_area_ratio
                    0.07,  # jawline_angle
                    0.07,  # chin_ratio
                    0.07,  # symmetry_score
                    0.15,  # skin_smoothness (increased for females)
                    0.07,  # eyebrow_texture
                    0.07,  # cheekbone_texture
                    0.07,  # forehead_texture
                    0.07,  # hairline_straightness
                    0.15   # lip_texture (increased for females)
                ])
                
                weighted_features = gender_features * feature_weights
                
                # Print weighted feature scores
                print("\nWeighted Feature Scores:")
                print(f"Face ratio score: {weighted_features[0]:.3f}")
                print(f"Eye ratio score: {weighted_features[1]:.3f}")
                print(f"Eye area ratio score: {weighted_features[2]:.3f}")
                print(f"Jawline angle score: {weighted_features[3]:.3f}")
                print(f"Chin ratio score: {weighted_features[4]:.3f}")
                print(f"Symmetry score: {weighted_features[5]:.3f}")
                print(f"Skin smoothness score: {weighted_features[6]:.3f}")
                print(f"Eyebrow texture score: {weighted_features[7]:.3f}")
                print(f"Cheekbone texture score: {weighted_features[8]:.3f}")
                print(f"Forehead texture score: {weighted_features[9]:.3f}")
                print(f"Hairline straightness score: {weighted_features[10]:.3f}")
                print(f"Lip texture score: {weighted_features[11]:.3f}")
                
                # College student specific adjustments
                if 18 <= self.get_age_value(self.prev_age) <= 25:  # College age range
                    print("\nCollege Student Adjustments:")
                    # Adjust weights for college students
                    if skin_smoothness > 0.7:  # Smooth skin is common in college students
                        weighted_features[6] *= 1.5  # Increase weight of skin smoothness
                        print("Applied skin smoothness boost")
                    
                    if eye_area_ratio > 0.8:  # More symmetrical eyes
                        weighted_features[2] *= 1.3  # Increase weight of eye area ratio
                        print("Applied eye area ratio boost")
                    
                    if symmetry_score < 0.3:  # High facial symmetry
                        weighted_features[5] *= 1.3  # Increase weight of symmetry
                        print("Applied symmetry boost")
                    
                    # Additional adjustments for college girls
                    if lip_texture < 100:  # Smooth lips
                        weighted_features[11] *= 1.5  # Increase weight of lip texture
                        print("Applied lip texture boost")
                    
                    if hairline_straightness > 0.5:  # Straight hairline
                        weighted_features[10] *= 1.3  # Increase weight of hairline
                        print("Applied hairline boost")
                    
                    if forehead_texture < 100:  # Smooth forehead
                        weighted_features[9] *= 1.3  # Increase weight of forehead texture
                        print("Applied forehead texture boost")
                    
                    # Strong female bias for college age
                    gender_preds[0][1] += 0.3  # Increase female probability
                    print("Applied college age female boost")
                
                # Adjust gender prediction based on features
                if np.sum(weighted_features) > 0:
                    gender_preds[0][1] += 0.2  # Increase female probability
                    print("Applied female feature boost")
                else:
                    gender_preds[0][0] += 0.2  # Increase male probability
                    print("Applied male feature boost")
            
            # Apply bias correction for female detection
            female_conf = gender_preds[0][1] + self.gender_bias_correction
            male_conf = gender_preds[0][0]
            
            # Normalize after bias correction
            total_conf = female_conf + male_conf
            female_conf = female_conf / total_conf
            male_conf = male_conf / total_conf
            
            # Determine gender with bias correction
            gender_conf = max(female_conf, male_conf)
            gender_idx = 1 if female_conf > male_conf else 0
            
            # Print final confidence scores
            print("\nFinal Gender Detection Results:")
            print(f"Male confidence: {male_conf:.3f}")
            print(f"Female confidence: {female_conf:.3f}")
            print(f"Selected gender: {self.gender_list[gender_idx]}")
            print(f"Confidence: {gender_conf:.3f}")
            
            # Apply temporal smoothing to gender
            self.gender_history.append((self.gender_list[gender_idx], gender_conf))
            if len(self.gender_history) > self.history_size:
                self.gender_history.pop(0)
            
            # Determine stable gender prediction with enhanced voting
            if len(self.gender_history) == self.history_size:
                gender_votes = {}
                for g, conf in self.gender_history:
                    if conf > self.gender_confidence_threshold:
                        gender_votes[g] = gender_votes.get(g, 0) + conf
                
                if gender_votes:
                    gender = max(gender_votes.items(), key=lambda x: x[1])[0]
                    print(f"\nStable gender prediction: {gender}")
                else:
                    # If no confident votes, use weighted average
                    female_total = sum(conf for g, conf in self.gender_history if g == 'Female')
                    male_total = sum(conf for g, conf in self.gender_history if g == 'Male')
                    
                    if female_total > male_total:
                        gender = 'Female'
                    else:
                        gender = 'Male'
                    print(f"\nWeighted average gender prediction: {gender}")
            else:
                gender = self.prev_gender if self.prev_gender else self.gender_list[gender_idx]
                print(f"\nUsing previous gender prediction: {gender}")
            
            self.prev_gender = gender
            
            # Age detection with enhanced accuracy
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            
            # Get weighted age prediction
            weighted_age = self.calculate_weighted_age(age_preds[0], face_img)
            
            # Store age prediction in history
            self.age_history.append(weighted_age)
            if len(self.age_history) > self.history_size:
                self.age_history.pop(0)
            
            # Apply temporal smoothing to age with outlier rejection
            if len(self.age_history) == self.history_size:
                # Calculate moving average with outlier rejection
                sorted_ages = sorted(self.age_history)
                # Remove potential outliers (top and bottom 10%)
                trimmed_ages = sorted_ages[1:-1]  # Remove one from each end
                smoothed_age = sum(trimmed_ages) / len(trimmed_ages)
                
                # If the change is too drastic, use exponential smoothing
                if self.prev_age is not None:
                    prev_age_value = self.get_age_value(self.prev_age)
                    if abs(smoothed_age - prev_age_value) > 5:
                        smoothed_age = 0.7 * prev_age_value + 0.3 * smoothed_age
                
                age = self.convert_age_to_range(smoothed_age)
            else:
                age = self.convert_age_to_range(weighted_age)
            
            self.prev_age = age
            
            # Print age detection details
            print("\nAge Detection Details:")
            print(f"Weighted age: {weighted_age:.1f}")
            print(f"Age range: {age}")
            print(f"Previous age: {self.prev_age}")
            
            return age, gender
        except Exception as e:
            print(f"Error in age/gender detection: {str(e)}")
            return self.prev_age if self.prev_age else "Unknown", self.prev_gender if self.prev_gender else "Unknown"
    
    def get_age_value(self, age_range):
        """Convert age range to numeric value."""
        if age_range == "Unknown":
            return 25  # Default middle age
        if '-' in age_range:
            min_age, max_age = map(int, age_range.split('-'))
            return (min_age + max_age) / 2
        return int(age_range.replace('+', ''))
    
    def convert_age_to_range(self, age_value):
        """Convert numeric age to age range with enhanced accuracy."""
        try:
            # Define age ranges with more granularity
            age_ranges = [
                (0, 2, '0-2'),
                (4, 6, '4-6'),
                (8, 12, '8-12'),
                (15, 20, '15-20'),
                (21, 25, '21-25'),
                (26, 32, '26-32'),
                (33, 40, '33-40'),
                (41, 50, '41-50'),
                (51, 60, '51-60'),
                (61, 100, '60+')
            ]
            
            # Find the appropriate age range
            for min_age, max_age, range_str in age_ranges:
                if min_age <= age_value <= max_age:
                    return range_str
            
            # If age is outside defined ranges, return closest range
            if age_value < 0:
                return '0-2'
            elif age_value > 100:
                return '60+'
            else:
                # Find closest range
                closest_range = min(age_ranges, key=lambda x: abs((x[0] + x[1])/2 - age_value))
                return closest_range[2]
        except Exception as e:
            print(f"Error in age range conversion: {str(e)}")
            return self.prev_age if self.prev_age else "Unknown"
    
    def detect_emotion(self, face_img):
        """Enhanced emotion detection with improved accuracy for neutral vs fear."""
        try:
            # Enhanced preprocessing
            face_img = self.preprocess_face(face_img)
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            gray_face_resized = cv2.resize(gray_face, (48, 48))
            
            # Apply histogram equalization for better contrast
            gray_face_resized = cv2.equalizeHist(gray_face_resized)
            
            # Apply additional preprocessing for better emotion detection
            gray_face_resized = cv2.GaussianBlur(gray_face_resized, (3,3), 0)
            
            # Normalize and reshape
            img = np.array(gray_face_resized).reshape(1, 48, 48, 1) / 255.0
            
            # Get predictions
            pred = self.emotion_model.predict(img, verbose=0)
            pred_idx = np.argmax(pred)
            confidence = pred[0][pred_idx]
            
            # Apply emotion-specific confidence thresholds
            emotion_thresholds = {
                0: 0.5,  # Angry - increased threshold
                1: 0.4,  # Disgust
                2: 0.5,  # Fear - increased threshold
                3: 0.4,  # Happy
                4: 0.4,  # Neutral
                5: 0.4,  # Sad
                6: 0.4   # Surprise
            }
            
            # Enhanced temporal smoothing for emotions
            self.emotion_history.append((pred_idx, confidence))
            if len(self.emotion_history) > self.history_size:
                self.emotion_history.pop(0)
            
            # Initialize emotion variable
            emotion = None
            
            # Apply temporal smoothing with confidence weighting
            if len(self.emotion_history) == self.history_size:
                emotion_votes = {}
                for idx, conf in self.emotion_history:
                    if conf > emotion_thresholds[idx]:
                        emotion_votes[idx] = emotion_votes.get(idx, 0) + conf
                
                if emotion_votes:
                    stable_idx = max(emotion_votes.items(), key=lambda x: x[1])[0]
                    emotion = self.emotion_labels[stable_idx]
                else:
                    # If no confident votes, use weighted average
                    emotion_scores = np.zeros(7)
                    for idx, conf in self.emotion_history:
                        emotion_scores[idx] += conf
                    
                    # Apply additional weighting for neutral vs fear
                    if emotion_scores[2] > 0.3 and emotion_scores[4] > 0.3:  # If both fear and neutral are high
                        # Check facial features to distinguish
                        face_features = self.analyze_facial_features(face_img)
                        if face_features['is_neutral']:
                            emotion_scores[4] *= 1.5  # Boost neutral score
                        else:
                            emotion_scores[2] *= 1.5  # Boost fear score
                    
                    emotion = self.emotion_labels[np.argmax(emotion_scores)]
            
            # If no emotion determined from history, use current prediction
            if emotion is None:
                if confidence > emotion_thresholds[pred_idx]:
                    emotion = self.emotion_labels[pred_idx]
                else:
                    emotion = self.prev_emotion if self.prev_emotion else "Neutral"
            
            self.prev_emotion = emotion
            
            # Return both emotion and confidence scores
            return emotion, pred[0]
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return self.prev_emotion if self.prev_emotion else "Unknown", None
    
    def analyze_facial_features(self, face_img):
        """Analyze facial features to distinguish between neutral and fear."""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Detect facial landmarks
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = face_img[y:y+h, x:x+w]
                
                # Detect eyes
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Detect mouth
                mouth = mouth_cascade.detectMultiScale(roi_gray)
                
                # Calculate features
                features = {
                    'is_neutral': True,
                    'eye_spacing': 0,
                    'mouth_width': 0,
                    'eyebrow_angle': 0,
                    'eye_openness': 0
                }
                
                # Analyze eye spacing
                if len(eyes) >= 2:
                    eye_spacing = abs(eyes[0][0] - eyes[1][0])
                    features['eye_spacing'] = eye_spacing / w
                    
                    # Calculate eye openness
                    eye_areas = []
                    for (ex, ey, ew, eh) in eyes:
                        eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
                        eye_areas.append(ew * eh)
                    features['eye_openness'] = min(eye_areas) / max(eye_areas)
                
                # Analyze mouth
                if len(mouth) > 0:
                    mouth_width = mouth[0][2]
                    features['mouth_width'] = mouth_width / w
                
                # Analyze eyebrow region
                eyebrow_region = roi_gray[max(0, y-20):y, x:x+w]
                if eyebrow_region.size > 0:
                    eyebrow_edges = cv2.Canny(eyebrow_region, 50, 150)
                    eyebrow_angle = np.arctan2(np.sum(eyebrow_edges), eyebrow_edges.shape[1])
                    features['eyebrow_angle'] = abs(eyebrow_angle)
                
                # Determine if expression is neutral
                if (features['eye_spacing'] > 0.2 and  # Normal eye spacing
                    features['mouth_width'] < 0.4 and  # Not wide mouth
                    features['eyebrow_angle'] < 0.3 and  # Not raised eyebrows
                    features['eye_openness'] > 0.7):  # Eyes not wide open
                    features['is_neutral'] = True
                else:
                    features['is_neutral'] = False
                
                return features
            
            return {'is_neutral': True}  # Default to neutral if no features detected
            
        except Exception as e:
            print(f"Error in facial feature analysis: {str(e)}")
            return {'is_neutral': True}  # Default to neutral on error
    
    def process_frame(self, frame):
        """Process a frame for detection with enhanced visualization."""
        try:
            faces = self.detect_face(frame)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                age, gender = self.detect_age_gender(face_img)
                emotion, confidence_scores = self.detect_emotion(face_img)
                
                # Get voice emotion if available
                voice_emotion = self.prev_voice_emotion
                
                # Draw rectangle and text with enhanced visualization
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Create a semi-transparent overlay for text
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y - 80), (x + w, y), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Display predictions with confidence scores
                cv2.putText(frame, f'Age: {age}', (x + 5, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f'Gender: {gender}', (x + 5, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f'Face Emotion: {emotion}', (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                if voice_emotion:
                    cv2.putText(frame, f'Voice Emotion: {voice_emotion}', (x + 5, y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                # Draw confidence scores as a bar chart with improved visualization
                if confidence_scores is not None:
                    bar_width = 30
                    bar_spacing = 5
                    start_x = x
                    start_y = y + h + 10
                    max_height = 50
                    
                    for i, (emotion_name, score) in enumerate(self.emotion_labels.items()):
                        bar_height = int(score * max_height)
                        color = (0, int(255 * score), 0)
                        cv2.rectangle(frame, 
                                    (start_x, start_y + max_height - bar_height),
                                    (start_x + bar_width, start_y + max_height),
                                    color, -1)
                        cv2.putText(frame, f'{emotion_name}: {score:.2f}',
                                  (start_x, start_y + max_height + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        start_x += bar_width + bar_spacing
            return frame
        except Exception as e:
            print(f"Error in frame processing: {str(e)}")
            return frame
    
    def calculate_weighted_age(self, age_preds, face_img):
        """Calculate weighted age prediction with enhanced accuracy."""
        try:
            # Check if face belongs to a child
            is_child_face = self.is_child(face_img)
            
            # Calculate weighted age prediction
            total_conf = sum(age_preds)
            if total_conf > 0:
                weighted_age = 0
                age_weights = np.zeros(len(self.age_list))
                
                # Print raw predictions for debugging
                print("\nRaw age predictions:")
                for idx, conf in enumerate(age_preds):
                    print(f"{self.age_list[idx]}: {conf:.4f}")
                
                # Apply age-specific weights with enhanced accuracy
                for idx, conf in enumerate(age_preds):
                    age_range = self.age_list[idx]
                    if '-' in age_range:
                        min_age, max_age = map(int, age_range.split('-'))
                        age_value = (min_age + max_age) / 2
                    else:
                        age_value = int(age_range.replace('+', ''))
                    
                    # Adjust weights based on age range and child detection
                    if is_child_face:
                        if age_value <= 12:  # Children
                            weight_factor = 3.5  # Increased weight for children
                        else:
                            weight_factor = 0.2  # Decreased weight for non-children
                    else:
                        if age_value <= 12:  # Children
                            weight_factor = 0.2  # Decreased weight for children
                        elif 21 <= age_value <= 25:  # College age range
                            weight_factor = 2.0  # Increased weight for college age
                        else:
                            weight_factor = 1.0  # Normal weight
                    
                    # Additional weight adjustment based on confidence
                    if conf > self.age_confidence_threshold:
                        weight_factor *= 1.5  # Boost high confidence predictions
                    
                    # Special handling for 21-25 age range
                    if age_range == '21-25':
                        # Increase weight if previous prediction was in this range
                        if self.prev_age == '21-25':
                            weight_factor *= 1.3
                        # Decrease weight for adjacent ranges
                        if age_range in ['15-20', '26-32']:
                            weight_factor *= 0.7
                    
                    age_weights[idx] = conf * weight_factor
                
                # Normalize weights and calculate final age
                total_weight = sum(age_weights)
                if total_weight > 0:
                    for idx, weight in enumerate(age_weights):
                        age_range = self.age_list[idx]
                        if '-' in age_range:
                            min_age, max_age = map(int, age_range.split('-'))
                            age_value = (min_age + max_age) / 2
                        else:
                            age_value = int(age_range.replace('+', ''))
                        weighted_age += age_value * (weight / total_weight)
                
                return weighted_age
            return 25
        except Exception as e:
            print(f"Error in age weight calculation: {str(e)}")
            return 25  # Default middle age

def video_stream():
    """Start the real-time video stream with voice emotion recognition."""
    webcam = None
    try:
        analyzer = FaceAnalyzer()
        webcam = cv2.VideoCapture(0)
        if not webcam.isOpened():
            raise Exception("Could not access webcam")
        
        print("\n🎥 Face and Voice Analysis System Started!")
        print("Press 'q' to exit.")
        print("Press 'r' to start/stop voice recording.")
        
        recording = False
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            processed_frame = analyzer.process_frame(frame)
            
            # Display frame
            cv2.imshow("Face and Voice Analysis", processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not recording:
                    print("Starting voice recording...")
                    analyzer.start_voice_recording()
                    recording = True
                else:
                    print("Stopping voice recording...")
                    analyzer.stop_voice_recording()
                    recording = False
            
    except Exception as e:
        print(f" Error: {str(e)}")
        print(traceback.format_exc())
    finally:
        # Cleanup
        if webcam is not None:
            webcam.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    video_stream()