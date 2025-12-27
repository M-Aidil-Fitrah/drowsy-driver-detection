"""
Alert System untuk Drowsiness Detection
"""

import pygame
import cv2
import os

class AlertSystem:
    def __init__(self, alert_sound_path="assets/alert.wav"):
        """
        Initialize alert system
        
        Args:
            alert_sound_path: Path ke file sound alert
        """
        pygame.mixer.init()
        
        # Load alert sound jika ada
        if os.path.exists(alert_sound_path):
            self.alert_sound = pygame.mixer.Sound(alert_sound_path)
        else:
            print(f"Warning: Alert sound not found at {alert_sound_path}")
            self.alert_sound = None
        
        self.is_alerting = False
    
    def trigger_visual_alert(self, frame, text="DROWSINESS ALERT!"):
        """
        Tampilkan visual alert di frame
        
        Args:
            frame: Video frame
            text: Alert message
            
        Returns:
            frame: Frame dengan alert overlay
        """
        # Overlay merah semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), 
                     (0, 0, 255), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Text alert
        cv2.putText(frame, text, (50, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        return frame
    
    def trigger_audio_alert(self):
        """Play alert sound"""
        if self.alert_sound and not self.is_alerting:
            self.alert_sound.play()
            self.is_alerting = True
    
    def stop_alert(self):
        """Stop alert"""
        if self.alert_sound:
            self.alert_sound.stop()
        self.is_alerting = False