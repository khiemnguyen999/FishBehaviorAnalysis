# Fish Behavior Analysis Using YOLO and DeepSORT

This repository provides a Google Colab implementation for automated fish behavior analysis using a deep learning-based object detection and tracking framework. The system detects fish in underwater videos, tracks individual fish across frames, and extracts multiple behavioral metrics, including swimming speed, turning angle, swimming direction, behavioral state, behavioral signature, and state transition patterns.

# Features

- Fish detection using YOLO (Ultralytics)
- Multi-object tracking using DeepSORT
- Image enhancement using CLAHE
- Swimming speed estimation
- Turning angle estimation
- Swimming state classification
- Swimming trajectory visualization
- Swimming direction analysis
- Rose diagram generation
- Behavioral signature extraction
- Markov transition graph generation


# Workflow

The complete analysis pipeline is illustrated below.

1. Install the required packages.
2. Mount Google Drive.
3. Load the pre-trained YOLO fish detection model.
4. Read the input video.
5. Enhance each frame using CLAHE.
6. Detect fish using YOLO.
7. Track fish using DeepSORT.
8. Estimate swimming speed and direction.
9. Classify swimming behavior.
10. Visualize fish trajectories.
11. Export behavioral metrics.
12. Generate behavioral statistics and visualizations.


# Installation

## Install Ultralytics

pip install ultralytics

## Install DeepSORT

pip install opencv-python-headless deep_sort_realtime

# Required Python Packages

- Python 3.10 or above
- Google Colab
- ultralytics
- OpenCV
- NumPy
- Pandas
- Matplotlib
- SciPy
- NetworkX
- Seaborn
- deep_sort_realtime

# Input Files

The program requires the following files.

## 1. Fish Detection Model

The folder **Model** contains the pre-trained fish detection model.

Please download **part1** and **part2**, combine them to create **model.zip**, and upload **model.zip** to your Google Drive before running the notebook.

The notebook will automatically extract the model during execution.


## 2. Fish Video

Provide an underwater fish video in MP4 format.

# Running the Notebook

Modify the following paths before execution.

video_path = "/content/drive/MyDrive/YourFolder/video.mp4"

model = YOLO("/content/dataset/.../best.pt")

After updating the paths, run all cells sequentially.

# Output Files

The program automatically generates the mp4 outputs.

# Swimming States

The swimming behavior is classified into five categories.

# Behavioral Signature

The generated behavioral signature includes

- Mean swimming speed
- Speed coefficient of variation
- Turning rate
- Directionality (R)
- Fast swimming ratio

# Notes

- The notebook is designed to run in Google Colab.
- Google Drive is used for loading the detection model and input videos.
- Please update the paths to your own Google Drive before execution.
- The pre-trained detection model is provided only for fish detection. Users may replace it with their own YOLO model if desired.


# Citation

If you use this repository in your research, please cite the associated publication.


# Contact

For questions or suggestions, please contact

**Your Name**

Email: nmkhiem@cit.ctu.edu.vn
