# Military Aircraft Detection using YOLOv10
## Overview
This project implements a YOLOv10-based object detection pipeline for detecting and classifying military aircraft from images.  
The pipeline covers dataset preprocessing, model training, evaluation, inference, and model export.

The project uses the Military Aircraft Detection Dataset from Kaggle, which includes 95 types of military aircraft with bounding box annotations.

## Dataset
* Source: Military Aircraft Detection Dataset (Kaggle: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)

* Description:

  * Fine-grained detection dataset with 95 aircraft types(It was 74 types when we worked on it).
  * Bounding box annotations in .csv files.
  * Includes both raw images and cropped samples.
    
## Project Workflow

1. Dataset Preprocessing

  * Converted .csv annotations → YOLO .txt format.
  * Organized into images/ and labels/ (train/val).
  * Created mydata.yaml for training.

2. Model Training

  * Used YOLOv10-small (yolov10s.pt) as pretrained model.
  * Trained for 100 epochs at 640×640 resolution.

3. Model Export

  * Exported to ONNX.
  * Saved weights as best_model.pt.
    
4. Inference

  * Run predictions on test images.
  * Draw bounding boxes + class labels with OpenCV.
  * Visualize results with Matplotlib.

## Results
* Model successfully trained to detect aircraft types across 95 categories.
* Predictions showed bounding boxes with class IDs.
* Example detections include F-35, Rafale, Su-25, Eurofighter Typhoon.

## Notes

* Due to GitHub’s 25 MB limit, the Jupyter Notebook (.ipynb) could not be uploaded (original size ~40 MB).

  * Outputs were cleared, but size was still too large.
  * Final solution: converted to PDF and uploaded instead.
 
## Future Work
* Improve dataset balance (some classes underrepresented).
* Hyperparameter tuning for better accuracy.
* Deploy trained model as an API or web app for real-time detection.

## References
* YOLOv10 GitHub (https://github.com/THU-MIG/yolov10)
* Ultralytics YOLO Docs (https://docs.ultralytics.com/)
* Military Aircraft Detection Dataset (https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)
