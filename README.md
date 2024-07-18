# YOLOv8 Fine-Tuning on Custom Dataset for Box Detection with Barcodes
**Overview**
This project involves gathering a custom dataset of boxes with barcodes, annotating the images, and fine-tuning the YOLOv8 model for accurate detection of boxes with barcodes. The project leverages OpenCV for image processing and Streamlit for the front-end interface.

## Technologies Used
- **YOLOv8**: For object detection model fine-tuning.
- **OpenCV**: For image processing and handling.
- **Streamlit**: For creating an interactive front-end interface.

## Preview
### <video width="1000" controls>
  <source src="preview\box.mp4" type="video/mp4">
    </video>

### <img src="preview\pred_1.png" width="1000"/>

### <img src="preview\pred_2.png" width="1000"/>

### <img src="preview\pred_3.png" width="1000"/>


## Features
- Custom Dataset Creation: Gathered and annotated a dataset of boxes with barcodes.
- YOLOv8 Fine-Tuning: Fine-tuned the YOLOv8 model on the custom dataset.
- Box and Barcode Detection: Used the fine-tuned model to detect boxes with barcodes.
- Interactive UI: User-friendly interface for uploading images and viewing detection results via Streamlit.

## Installation
Prerequisites
- Python 3.8 or higher
- Virtual environment tools (e.g., venv or virtualenv)

## Steps
Clone the Repository

```bash
git clone https://github.com/jagdish31502/box-detection.git
cd box-detection
```

Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage
Run the Streamlit App

``` bash
streamlit run app.py
```
- Upload Images
- Navigate to the Streamlit app in your browser.
- Upload images containing boxes with barcodes for detection.
- View Detection Results in runs/detect/predict