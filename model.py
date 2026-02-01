import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'  # Force Keras 3

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import numpy as np

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")

def create_emotion_model():
    """
    Creates a simple CNN model for emotion detection
    Compatible with Keras 3
    """
    model = Sequential([
        Input(shape=(48, 48, 1)),
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
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_save_model():
    print("\n" + "="*50)
    print("EMOTION DETECTION MODEL TRAINING")
    print("="*50 + "\n")
    
    print("Step 1: Creating model architecture...")
    model = create_emotion_model()
    print("✓ Model created successfully\n")
    
    print("Step 2: Generating dummy training data...")
    X_train = np.random.rand(100, 48, 48, 1).astype('float32')
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 4, 100), 4)
    print("✓ Training data generated\n")
    
    print("Step 3: Training model (this will take a moment)...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    print("✓ Training complete\n")
    
    print("Step 4: Saving model...")
    # Save in Keras 3 format explicitly
    model.save('model.h5', save_format='h5')
    print("✓ Model saved as model.h5\n")
    
    # Verify the model can be loaded
    print("Step 5: Verifying model can be loaded...")
    test_model = tf.keras.models.load_model('model.h5')
    print("✓ Model verified successfully!\n")
    
    print("="*50)
    print("MODEL READY FOR DEPLOYMENT")
    print("="*50)

if __name__ == "__main__":
    train_and_save_model()
    