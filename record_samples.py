import sounddevice as sd
import soundfile as sf
import os
import time

# Create voice_data directory if it doesn't exist
if not os.path.exists('voice_data'):
    os.makedirs('voice_data')

# Emotions to record
emotions = ['Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Disgust', 'Surprise']

# Recording parameters
duration = 3  # seconds
sample_rate = 22050

def record_sample(emotion, sample_number):
    print(f"\nRecording {emotion} sample {sample_number}")
    print("Get ready...")
    time.sleep(1)
    print("Recording in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Speak now!")
    
    # Record audio
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    
    # Create emotion directory if it doesn't exist
    emotion_dir = os.path.join('voice_data', emotion)
    if not os.path.exists(emotion_dir):
        os.makedirs(emotion_dir)
    
    # Save the recording
    filename = os.path.join(emotion_dir, f'{emotion}_{sample_number}.wav')
    sf.write(filename, recording, sample_rate)
    print(f"Saved to {filename}")

def main():
    print("Voice Sample Recording Tool")
    print("=========================")
    print("This tool will help you record voice samples for each emotion.")
    print("For each emotion, you'll record multiple samples.")
    print("Speak clearly and try to convey the emotion through your voice.")
    print("\nPress Enter to start recording samples...")
    input()
    
    samples_per_emotion = 5  # Number of samples to record for each emotion
    
    for emotion in emotions:
        print(f"\n=== Recording {emotion} samples ===")
        for i in range(samples_per_emotion):
            record_sample(emotion, i+1)
            print("\nPress Enter to continue to next sample...")
            input()
        
        print(f"\nCompleted recording {emotion} samples!")
        print("Press Enter to move to next emotion...")
        input()

if __name__ == "__main__":
    try:
        main()
        print("\nAll samples recorded successfully!")
        print("Now you can run 'python train_voice_model.py' to train the model")
    except KeyboardInterrupt:
        print("\nRecording interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 