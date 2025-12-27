"""
=============================================================================
DROWSY DRIVER DETECTION SYSTEM
=============================================================================
Sistem deteksi kantuk pengemudi menggunakan Vision Transformer (ViT)

Topik Computer Vision yang Tercakup:
1. Object Detection - Face detection menggunakan Haar Cascade
2. Object Tracking - Tracking wajah frame-by-frame
3. Object Recognition - Klasifikasi drowsy/not drowsy menggunakan ViT
4. CNN/Transformers - Vision Transformer (ViT-Base) architecture

Model Information:
- Architecture: Vision Transformer (ViT-Base)
- Accuracy: 97.52%
- Dataset: UTA-RLDD (Real-Life Drowsiness Dataset)
- Classes: drowsy, not drowsy
=============================================================================
"""

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import time
import os
import sys

# Import utils
from utils.face_detector import FaceDetector
from utils.preprocessor import ImagePreprocessor
from utils.alert_system import AlertSystem


class DrowsyDriverDetector:
    """
    Main class untuk Drowsy Driver Detection System
    """
    
    def __init__(self, model_path="./models", alert_sound_path="./assets/alert.wav"):
        """
        Inisialisasi detector
        
        Args:
            model_path: Path ke folder model ViT
            alert_sound_path: Path ke file alert sound
        """
        print("=" * 70)
        print("DROWSY DRIVER DETECTION SYSTEM")
        print("=" * 70)
        print("\n🔧 Initializing system...")
        
        # Setup device (GPU jika tersedia, kalau tidak pakai CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Device: {self.device}")
        
        # Load ViT Model
        print(f"🤖 Loading ViT model from {model_path}...")
        try:
            self.processor = ViTImageProcessor.from_pretrained(model_path)
            self.model = ViTForImageClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
            print(f"   Classes: {self.model.config.id2label}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
        
        # Initialize components
        print("\n🔧 Initializing components...")
        self.face_detector = FaceDetector()
        self.preprocessor = ImagePreprocessor(self.processor)
        self.alert_system = AlertSystem(alert_sound_path)
        print("✅ All components initialized!")
        
        # Detection parameters
        self.drowsy_counter = 0  # Counter untuk tracking drowsiness
        self.drowsy_threshold = 15  # Alert jika drowsy 15 frames berturut-turut (~0.5 detik at 30 FPS)
        self.prediction_interval = 3  # Predict setiap 3 frame (untuk performa)
        self.frame_count = 0
        
        # Statistics
        self.total_frames = 0
        self.drowsy_detections = 0
        self.alert_triggered_count = 0
        
        # Last prediction cache
        self.last_prediction = None
        self.last_confidence = 0.0
        
        print("\n✅ System ready!")
        print("=" * 70)
    
    def predict_drowsiness(self, face_img):
        """
        Predict drowsiness dari face image menggunakan ViT model
        
        Args:
            face_img: Face image (BGR format dari OpenCV)
            
        Returns:
            label: 'drowsy' atau 'notdrowsy'
            confidence: Confidence score (0-1)
        """
        try:
            # Preprocess image
            inputs = self.preprocessor.preprocess(face_img)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Get prediction
            predicted_class = logits.argmax(-1).item()
            probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
            confidence = probabilities[predicted_class].item()
            
            label = self.model.config.id2label[str(predicted_class)]
            
            return label, confidence
            
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return None, 0.0
    
    def draw_info(self, frame, face_coords, label, confidence, fps):
        """
        Gambar informasi di frame (bounding box, label, statistik)
        
        Args:
            frame: Video frame
            face_coords: Tuple (x, y, w, h) koordinat face
            label: Prediction label
            confidence: Confidence score
            fps: Frame per second
            
        Returns:
            frame: Frame dengan overlay informasi
        """
        height, width = frame.shape[:2]
        
        # Draw bounding box pada wajah
        if face_coords is not None:
            (x, y, w, h) = face_coords
            
            # Warna box: merah jika drowsy, hijau jika awake
            color = (0, 0, 255) if label == "drowsy" else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Label prediction di atas bounding box
            label_text = f"{label.upper()}: {confidence:.1%}"
            cv2.putText(frame, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Info panel (kiri atas)
        info_y = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_y += 30
        cv2.putText(frame, f"Total Frames: {self.total_frames}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(frame, f"Drowsy Count: {self.drowsy_detections}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        info_y += 25
        cv2.putText(frame, f"Alerts: {self.alert_triggered_count}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Status indicator (kanan atas)
        status_text = "STATUS: DROWSY!" if label == "drowsy" else "STATUS: AWAKE"
        status_color = (0, 0, 255) if label == "drowsy" else (0, 255, 0)
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        status_x = width - text_size[0] - 10
        cv2.putText(frame, status_text, (status_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Drowsiness meter (progress bar)
        if label == "drowsy":
            bar_width = 200
            bar_height = 20
            bar_x = width - bar_width - 10
            bar_y = 50
            
            # Background bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (100, 100, 100), -1)
            
            # Progress bar (berdasarkan drowsy_counter)
            progress = min(self.drowsy_counter / self.drowsy_threshold, 1.0)
            progress_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                         (0, 0, 255), -1)
            
            # Text
            cv2.putText(frame, f"Alert in: {max(0, self.drowsy_threshold - self.drowsy_counter)}", 
                       (bar_x, bar_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, video_source=0, output_path=None, save_log=True):
        """
        Jalankan deteksi drowsiness secara real-time
        
        Args:
            video_source: Camera index (0 untuk webcam default) atau path video file
            output_path: Path untuk save output video (optional)
            save_log: Simpan log deteksi ke CSV (optional)
        """
        print("\n🎥 Starting video capture...")
        print("Press 'q' to quit, 'r' to reset statistics")
        print("=" * 70)
        
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open camera/video!")
            return
        
        # Setup video writer jika output_path diberikan
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_out = int(cap.get(cv2.CAP_PROP_FPS))
            width_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(output_path, fourcc, fps_out, (width_out, height_out))
            print(f"📹 Saving output to: {output_path}")
        
        # Setup log file jika save_log True
        if save_log:
            log_file = open("data/drowsy_log.csv", "w")
            log_file.write("timestamp,frame,label,confidence,drowsy_counter,alert\n")
            print(f"📝 Saving log to: data/drowsy_log.csv")
        
        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        print("\n▶️  Detection started!\n")
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ End of video or cannot read frame")
                    break
                
                self.total_frames += 1
                self.frame_count += 1
                fps_frame_count += 1
                
                # Calculate FPS setiap 1 detik
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Detect face
                face_img, face_coords = self.face_detector.detect(frame)
                
                # Jika wajah terdeteksi
                if face_img is not None:
                    # Predict drowsiness setiap prediction_interval frames
                    if self.frame_count % self.prediction_interval == 0:
                        label, confidence = self.predict_drowsiness(face_img)
                        
                        if label is not None:
                            self.last_prediction = label
                            self.last_confidence = confidence
                    else:
                        # Gunakan prediksi terakhir
                        label = self.last_prediction
                        confidence = self.last_confidence
                    
                    # Update drowsy counter dan alert logic
                    if label == "drowsy":
                        self.drowsy_counter += 1
                        self.drowsy_detections += 1
                        
                        # Trigger alert jika melewati threshold
                        if self.drowsy_counter >= self.drowsy_threshold:
                            # Visual alert
                            frame = self.alert_system.trigger_visual_alert(
                                frame, 
                                text="⚠️ DROWSINESS DETECTED! ⚠️"
                            )
                            
                            # Audio alert
                            self.alert_system.trigger_audio_alert()
                            
                            # Increment alert counter
                            if self.drowsy_counter == self.drowsy_threshold:
                                self.alert_triggered_count += 1
                                print(f"🚨 ALERT #{self.alert_triggered_count} - Drowsiness detected at frame {self.total_frames}!")
                    else:
                        # Reset counter jika awake
                        if self.drowsy_counter > 0:
                            self.drowsy_counter = 0
                            self.alert_system.stop_alert()
                    
                    # Draw information
                    frame = self.draw_info(frame, face_coords, label, confidence, fps)
                    
                    # Log to CSV
                    if save_log:
                        alert_status = "YES" if self.drowsy_counter >= self.drowsy_threshold else "NO"
                        log_file.write(f"{time.time()},{self.total_frames},{label},{confidence:.4f},{self.drowsy_counter},{alert_status}\n")
                
                else:
                    # Tidak ada wajah terdeteksi
                    cv2.putText(frame, "No face detected", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Reset counter jika tidak ada wajah
                    if self.drowsy_counter > 0:
                        self.drowsy_counter = 0
                        self.alert_system.stop_alert()
                
                # Write frame to output video jika ada
                if video_writer:
                    video_writer.write(frame)
                
                # Show frame
                cv2.imshow('Drowsy Driver Detection System', frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  Stopping detection...")
                    break
                elif key == ord('r'):
                    print("\n🔄 Resetting statistics...")
                    self.drowsy_counter = 0
                    self.drowsy_detections = 0
                    self.alert_triggered_count = 0
                    self.total_frames = 0
        
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user")
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            cap.release()
            if video_writer:
                video_writer.release()
            if save_log:
                log_file.close()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print("\n" + "=" * 70)
            print("DETECTION SUMMARY")
            print("=" * 70)
            print(f"Total Frames Processed: {self.total_frames}")
            print(f"Drowsy Detections: {self.drowsy_detections}")
            print(f"Alerts Triggered: {self.alert_triggered_count}")
            if self.total_frames > 0:
                drowsy_percentage = (self.drowsy_detections / self.total_frames) * 100
                print(f"Drowsiness Rate: {drowsy_percentage:.2f}%")
            print("=" * 70)
            print("✅ Detection completed!")


def main():
    """
    Main function
    """
    # Paths
    model_path = "./models"
    alert_sound_path = "./assets/alert.wav"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model folder not found at {model_path}")
        print("Please download the model first!")
        return
    
    # Initialize detector
    try:
        detector = DrowsyDriverDetector(
            model_path=model_path,
            alert_sound_path=alert_sound_path
        )
        
        # Run detection
        # video_source: 0 untuk webcam, atau path ke video file
        # output_path: None untuk tidak save, atau path output video
        detector.run(
            video_source=0,  # Webcam
            output_path=None,  # Tidak save video (set path jika mau save)
            save_log=True  # Save log ke CSV
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()