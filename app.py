import streamlit as st
from PIL import Image
import numpy as np
import sqlite3
from deepface import DeepFace

# Function to create the SQLite database connection
def create_connection():
    conn = sqlite3.connect('users.db')
    return conn

# Function to initialize the database table
def initialize_db():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            embedding BLOB
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert a new user into the database
def insert_user(name, embedding):
    conn = create_connection()
    c = conn.cursor()
    c.execute('INSERT INTO users (name, embedding) VALUES (?, ?)',
              (name, embedding))
    conn.commit()
    conn.close()

# Function to recognize faces using deepface and perform face analysis
def analyze_faces(image):
    try:
        faces = DeepFace.analyze(np.array(image), actions=['age', 'gender', 'emotion'])
        return faces
    except Exception as e:
        st.error(f"Error during face analysis: {e}")
        return None

def main():
    st.title("Face Recognition and Analysis")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Perform face analysis using deepface
        faces = analyze_faces(image)

        if faces is not None and len(faces) > 0:
            for idx, face in enumerate(faces):
                st.subheader(f"Face {idx + 1} Analysis:")
                st.write(f"Age: {face['age']}")
                st.write(f"Gender: {face['gender']}")
                st.write(f"Emotion: {face['dominant_emotion']}")

                # Perform face recognition logic here (compare with database, prompt for user info, etc.)

if __name__ == "__main__":
    main()
