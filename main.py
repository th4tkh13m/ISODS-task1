from task1.model import YOLOModel


if __name__ == "__main__":
    model = YOLOModel(model_name="yolov8x", model_path="yolov8x.pt")
    model.predict_images("imgs")
    model.to_json("detected_objs.json")