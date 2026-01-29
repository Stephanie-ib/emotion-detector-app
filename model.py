import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import numpy as np

def create_emotion_model():
    """
    Creates a simple CNN model for emotion detection
    Input: 48x48 grayscale images
    Output: 4 emotions (Happy, Sad, Angry, Neutral)
    """
    model = Sequential([
        Input(shape=(48, 48, 1)),  # Fixed for Keras 3
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_and_save_model():
    """
    Train the model with dummy data and save it
    """
    print("Creating model...")
    model = create_emotion_model()
    
    print("Generating dummy training data...")
    X_train = np.random.rand(100, 48, 48, 1)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 4, 100), 4)
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    
    print("Saving model...")
    model.save('model.h5')
    print("Model saved as model.h5 (Keras 3 compatible)")

if __name__ == "__main__":
    train_and_save_model()
    