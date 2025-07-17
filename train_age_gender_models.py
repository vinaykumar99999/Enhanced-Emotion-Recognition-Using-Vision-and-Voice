import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
import os

def create_age_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output age normalized between 0-1
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def create_gender_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Output gender probability
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

def main():
    # Create models
    age_model = create_age_model()
    gender_model = create_gender_model()
    
    # Save models
    age_model.save('age_model.h5')
    gender_model.save('gender_model.h5')
    print("Models saved successfully")

if __name__ == "__main__":
    main() 