import os
os.environ['TF_USE_LEGACY_KERAS'] = '0'  

from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('model.h5')

# Emotion labels
EMOTIONS = ['Angry', 'Happy', 'Neutral', 'Sad']

# Initialize database
def init_db():
    conn = sqlite3.connect('emotion_db.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            predicted_emotion TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def preprocess_image(image_path):
    """
    Preprocess the uploaded image for model prediction
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((48, 48))  # Resize to 48x48
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 48, 48, 1)  # Reshape for model
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'username' not in request.form:
        return redirect(url_for('index'))
    
    file = request.files['file']
    username = request.form['username']
    
    if file.filename == '' or username == '':
        return redirect(url_for('index'))
    
    if file:
        # Save the uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        processed_image = preprocess_image(filepath)
        prediction = model.predict(processed_image)
        emotion_index = np.argmax(prediction)
        predicted_emotion = EMOTIONS[emotion_index]
        confidence = float(prediction[0][emotion_index]) * 100
        
        # Save to database
        conn = sqlite3.connect('emotion_db.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (username, image_filename, predicted_emotion)
            VALUES (?, ?, ?)
        ''', (username, filename, predicted_emotion))
        conn.commit()
        conn.close()
        
        return render_template('index.html', 
                             prediction=predicted_emotion,
                             confidence=f"{confidence:.2f}",
                             username=username,
                             image_path=f"uploads/{filename}")
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
