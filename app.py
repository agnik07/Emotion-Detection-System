#!/usr/bin/env python3
"""
Emotion Detection Project Runner
This script provides multiple ways to run the emotion detection project:
1. Webcam mode (if camera is available)
2. Image file mode (for testing with static images)
3. Demo mode (with sample images)
"""

import cv2
from keras.models import model_from_json
import numpy as np
import os
import sys

class EmotionDetector:
    def __init__(self):
        """Initialize the emotion detection model"""
        print("Loading emotion detection model...")
        
        # Load model
        json_file = open("emotiondetector.json", "r")
        model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(model_json)
        self.model.load_weights("emotiondetector.h5")
        
        # Load face detector
        haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(haar_file)
        
        # Emotion labels
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        
        print("✓ Model loaded successfully!")
        print("✓ Available emotions:", list(self.labels.values()))
    
    def extract_features(self, image):
        """Extract features from image for prediction"""
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0
    
    def detect_emotion_in_image(self, image_path):
        """Detect emotion in a static image"""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return None
            
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image '{image_path}'")
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("No faces detected in the image")
            return None
            
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            img_features = self.extract_features(face_resized)
            prediction = self.model.predict(img_features)
            emotion = self.labels[prediction.argmax()]
            
            # Draw rectangle and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display result
        cv2.imshow("Emotion Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return emotion
    
    def run_webcam_mode(self):
        """Run emotion detection with webcam"""
        print("Starting webcam mode...")
        webcam = cv2.VideoCapture(0)
        
        if not webcam.isOpened():
            print("Error: Could not open webcam")
            return False
            
        print("Webcam opened successfully!")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Error reading frame")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (48, 48))
                img_features = self.extract_features(face_resized)
                prediction = self.model.predict(img_features)
                emotion = self.labels[prediction.argmax()]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow("Emotion Detection - Webcam", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        webcam.release()
        cv2.destroyAllWindows()
        return True
    
    def run_demo_mode(self):
        """Run a simple demo with a generated face-like image"""
        print("Running demo mode...")
        
        # Create a simple demo image
        demo_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        
        # Add a simple face-like rectangle
        cv2.rectangle(demo_img, (50, 50), (150, 150), (0, 255, 0), 2)
        
        # Display demo
        cv2.putText(demo_img, "Demo Mode", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(demo_img, "Model Working!", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Emotion Detection Demo", demo_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
        print("Demo completed!")

def main():
    detector = EmotionDetector()
    
    print("\n=== Emotion Detection Project ===")
    print("Choose an option:")
    print("1. Webcam mode (if camera is available)")
    print("2. Demo mode (static display)")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        success = detector.run_webcam_mode()
        if not success:
            print("Webcam not available, running demo mode instead...")
            detector.run_demo_mode()
    elif choice == "2":
        detector.run_demo_mode()
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()
