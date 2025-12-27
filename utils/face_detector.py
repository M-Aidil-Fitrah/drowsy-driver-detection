"""
Face Detection menggunakan Haar Cascade
"""

import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        """Initialize Haar Cascade face detector"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect(self, frame):
        """
        Deteksi wajah dari frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            face_img: Cropped face image
            coords: Tuple (x, y, w, h) koordinat face
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Ambil face pertama (terbesar)
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            return face_img, (x, y, w, h)
        
        return None, None