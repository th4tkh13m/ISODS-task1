import cv2
from random import uniform
import numpy as np
import os
from pydantic import BaseModel

class VideoProcessor(BaseModel):
    flip:int=1
    random_deg:float=10
    noise_mean:float=0
    noise_std:float=100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def random_rotate(self, frame, deg):
        random_deg = uniform(-deg, deg)

        h, w = frame.shape
        center = (w/2, h/2)

        rotation_matrix = cv2.getRotationMatrix2D(center, random_deg, 1.0)
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

        return rotated_frame

    def add_noise(self, frame, mean, std):
        noise = np.zeros_like(frame)
        cv2.randn(noise, mean, std)

        noisy_frame = cv2.add(frame, noise)
        return noisy_frame
    
    def flip_frame(self, frame, flip_code: int):
        flipped_frame = cv2.flip(frame, flip_code)
        return flipped_frame

    def process(self, vid_path: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(vid_path)
        index = 0

        if not cap.isOpened():
            exit(1)
        while True:
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = self.flip_frame(frame, self.flip)
                frame = self.random_rotate(frame, self.random_deg)
                frame = self.add_noise(frame, self.noise_mean, 
                                       self.noise_std)
                
                
                cv2.imwrite(f"{save_dir}/{index}.png", frame)

                index += 1
            else:
                break
        cap.release()