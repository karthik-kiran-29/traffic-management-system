## Prerequisites

- Python 3.6 or higher
- OpenCV
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/object-detection-project.git
   cd object-detection-project
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv env
   source env/bin/activate  # On Windows, use: env\Scripts\activate
   ```

3. Create a `.env` file in the project root (optional for environment variables)

4. Install the required dependencies:
   ```
   pip install opencv-python numpy
   ```

## Download YOLOv3 Files

Download the required YOLOv3 model files:

1. YOLOv3 weights file (237 MB):
   [Download yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

2. YOLOv3 configuration file:
   [Download yolov3.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)

Place these files in the project directory.

## Usage

Run the object detector script:

```
python object_detector.py
```
