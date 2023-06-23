from ultralytics import YOLO
import os
import json
from pydantic import BaseModel
from typing import Literal, Optional, Any
from task1.utils import get_img_list

BASE_LINK = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"

class YOLOModel(BaseModel):
    model_name: Literal["yolov8n",
                        "yolov8s",
                        "yolov8m",
                        "yolov8l",
                        "yolov8x"] = "yolov8m"
    model_path: Optional[str]
    model_save_dir: str = "."
    model: Any
    detected_objs: Any

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model_path is None:
            self.model_path = self.fetch_weights(self.model_name,
                                            self.model_save_dir)
        self.model = YOLO(self.model_path)
        self.detected_objs = []

    def fetch_weights(model_name: str,
                      save_dir: str):
        import requests

        model_path = BASE_LINK + model_name + ".pt"
        with requests.get(model_path) as response:

            if response.status_code != 200:
                raise ValueError("Cannot fetch the model.")
            
            os.makedirs(save_dir, exist_ok=True)

            file_path = os.path.join(save_dir, model_name + ".pt")
            with open(file_path, "wb") as f:
                f.write(response.content)
        return file_path
    
    def predict(self, src: Any):
        results = self.model(src)
        return results
    
    def _extract_data(self, results, detected_objs):
        for result in results:
            file_name = os.path.basename(result.path)
            frame_id = file_name.split(".png")[0]

            for box in result.boxes:
                bbox = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls = int(box.cls.item())

                data = {
                    "frame_id": frame_id,
                    "class": cls,
                    "bbox": bbox,
                    "confident": conf
                }

                detected_objs.append(data)

    def predict_images(self, img_dir: str):
        self.detected_objs = []

        img_files = get_img_list(img_dir)
        for f in img_files:
            file_path = os.path.join(img_dir, f)
            results = self.predict(file_path)
            self._extract_data(results, self.detected_objs)

    def to_json(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.detected_objs, f, indent=4)