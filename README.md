# Drone Rail Detection

This project is for **Computer Vision** and **Planning Techniques for Robotics**.

The goal is for a drone to detect railway features from a camera feed and create a path for navigation. The project uses the **Ultralytics YOLO workflow** for object detection and includes scripts for dataset preparation, model training, and inference.

It combines dataset preparation, model training, and inference scripts for drone and rail scene understanding.

## Team

This project was made by:
- [CataAlTreilea](https://github.com/CataAlTreilea)
- [VStefanTab](https://github.com/VStefanTab)
- [Balaciu9](https://github.com/Balaciu9)
- [sebastiankalciov](https://github.com/sebastiankalciov))



(Replace the placeholder names with your collaborators.)

## Technologies

- Python (core language)
- OpenCV (image and video processing)
- NumPy (numerical operations)
- Ultralytics YOLO workflow (training and detection pipeline)
- XML and NDJSON annotation formats (label handling)
- Robotics planning concepts for route and behavior support

## Project Structure

- auto_annotate.py: Utilities for annotation automation.
- convertToYaml.py: Converts dataset configuration for training.
- datasetTransformations.py: Augmentation and dataset transformation logic.
- extractFrames.py: Frame extraction from video/image sequences.
- rebalance_split.py: Dataset balancing and split helper.
- train.py: Model training entry script.
- main.py: Main execution flow for project logic.
- railDetection.py: Rail-related detection logic.
- droneControl.py: Robotics/drone control related script.
- path_preview.py: Path visualization and preview helper.
- requirements.txt: Python dependencies.

## Setup

1. Install Python 3.9 or newer.
2. Create and activate a virtual environment.
3. Install dependencies from requirements.txt.

Windows PowerShell example:

1. py -m venv .venv
2. pip install -r requirements.txt

## How To Run

Common entry points:

1. Train the model:

	python train.py

2. Run the main pipeline:

	python main.py

3. Run rail detection module directly:

	python railDetection.py

4. Extract frames from source data:

	python extractFrames.py

If your scripts require custom paths or runtime options, update the values inside each script or pass command-line arguments (where supported).

## Typical Workflow

1. Prepare and organize data in the Dataset folder.
2. Transform and rebalance data using datasetTransformations.py and rebalance_split.py.
3. Convert labels/configs with convertToYaml.py if needed.
4. Train with train.py.
5. Run inference and path/planning logic with main.py, railDetection.py, and path_preview.py.

## Outputs And Models

- Pretrained or experiment weights are stored as .pt files.
- Additional assets (for example flight path visuals) are included for testing and preview.

## Notes

The repository includes scripts and assets for data preparation, model training, and rail/drone detection experiments.
