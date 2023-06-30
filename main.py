import argparse
from task1.model import YOLOModel
from task1.processor import VideoProcessor

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Video Processing and Image Prediction')

    # Add command-line arguments
    parser.add_argument('--process-video', action='store_true', help='Process the video')
    parser.add_argument('--predict-images', action='store_true', help='Predict images')
    parser.add_argument('--vid-path', type=str, help='Path to the video')
    parser.add_argument('--img-save-dir', type=str, help='Directory to save the processed images')
    parser.add_argument('--model-name', type=str, default='yolov8m', help='Name of the model')
    parser.add_argument('--model-path', type=str, help='Path to the model')
    parser.add_argument('--json-path', type=str, help='Path to save the JSON output')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Extract the values from the arguments
    process_video = args.process_video
    predict_images = args.predict_images
    vid_path = args.vid_path
    img_save_dir = args.img_save_dir
    model_name = args.model_name
    model_path = args.model_path
    json_path = args.json_path

    # Video processing
    if process_video:
        processor = VideoProcessor()
        processor.process(vid_path, img_save_dir)

    # Image prediction
    if predict_images:
        model = YOLOModel(model_name, model_path)
        model.predict_images(img_save_dir)
        model.to_json(json_path)
