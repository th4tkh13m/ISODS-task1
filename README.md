# ISODS-task1

This project provides functionality for video processing and image prediction using a YOLO model.

## Description

This project leverages the YOLO (You Only Look Once) model to perform video processing and image prediction tasks.

The video processing functionality allows you to process a video file and extract individual frames as images. The image prediction functionality uses the YOLO model to perform object detection and prediction on the processed images.

## Installation

To install and set up the project, follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/your-repo.git`
2. Navigate to the project directory: `cd your-repo`
3. Install the required dependencies: `pip install -r requirements.txt`

Note: Make sure you have Python and pip installed on your system.

## Usage

To use the project, follow these instructions:

1. Ensure you have installed the project and its dependencies as mentioned in the installation section.
2. Prepare a video file that you want to process.
3. Open a terminal or command prompt and navigate to the project directory.
4. Run the main file with the desired command-line arguments:
   ```shell
   python main.py --process-video --predict-images --vid-path "path/to/video.mp4" --img-save-dir "path/to/save/images" --model-path "path/to/model" --json-path "path/to/save/json"
   ```

## JSON File format

```json
[
    {
        "frame_id": frame_id,
        "class": cls,
        "bbox": bbox,
        "confident": conf
    }
]
```
