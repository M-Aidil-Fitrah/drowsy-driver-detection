"""
Image Preprocessing untuk ViT Model
"""

import cv2
from PIL import Image

class ImagePreprocessor:
    def __init__(self, processor):
        """
        Args:
            processor: ViTImageProcessor dari transformers
        """
        self.processor = processor
    
    def preprocess(self, face_img):
        """
        Preprocess face image untuk ViT model
        
        Args:
            face_img: Face image (BGR format dari OpenCV)
            
        Returns:
            inputs: Tensor siap untuk model
        """
        # Convert BGR ke RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Convert ke PIL Image
        pil_image = Image.fromarray(face_rgb)
        
        # Preprocess dengan ViT processor
        inputs = self.processor(images=pil_image, return_tensors="pt")
        
        return inputs