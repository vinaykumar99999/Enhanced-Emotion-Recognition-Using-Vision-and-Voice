# Face and Voice Emotion Recognition System

This project implements a real-time face and voice emotion recognition system using machine learning. It can detect age, gender, facial emotions, and voice emotions from webcam input.

## Features

- Real-time face detection
- Age and gender prediction
- Facial emotion recognition (7 emotions)
- Voice emotion recognition
- Enhanced visualization with confidence scores
- Temporal smoothing for stable predictions
- College student-specific optimizations

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow/Keras
- NumPy
- Librosa
- SoundDevice
- SciPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Face_Emotion_Recognition_Machine_Learning.git
cd Face_Emotion_Recognition_Machine_Learning
```

2. Install required packages:

```bash
pip install -r requirements.txt
```

3. Download required model files:

- facialemotionmodel.json
- facialemotionmodel.h5
- age_net.caffemodel
- age_deploy.prototxt
- gender_net.caffemodel
- gender_deploy.prototxt

## Usage

Run the main script:

```bash
python realtimedetection.py
```

Controls:

- Press 'q' to exit
- Press 'r' to start/stop voice recording

## Project Structure

- `realtimedetection.py`: Main script containing the FaceAnalyzer class and video stream implementation
- Model files: Required for face, age, gender, and emotion detection
- Requirements file: Lists all Python dependencies

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
