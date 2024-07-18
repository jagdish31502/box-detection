import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
import random

# Function to save uploaded file
def save_uploaded_file(uploaded_file, save_path):
    with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return os.path.join(save_path, uploaded_file.name)

# Opening the file in read mode
my_file = open("utils/coco.txt", "r")
# Reading the file
data = my_file.read()
# Replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a model
model_path = os.path.join('.', 'weights', 'last.pt')
model = YOLO(model_path)  # load a custom model

# Streamlit App
st.title("Object Detection using YOLO")

# File upload and object detection
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file to a specific directory
    file_path = save_uploaded_file(uploaded_file, r"C:\Users\Evenmore\Downloads\learning\box_detection\utils\sample_images")

    # Read the image
    image = cv2.imread(file_path)

    # Check if the image was successfully opened
    if image is None:
        st.error("Could not open or find the image.")
    else:
        # Predict on image
        detect_params = model.predict(source=[image], conf=0.45, save=True)
        DP = detect_params[0].numpy()

        detected_classes = []
        if len(DP) != 0:
            for i in range(len(detect_params[0])):
                boxes = detect_params[0].boxes
                box = boxes[i]
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                # Append detected class and confidence to list
                detected_classes.append((class_list[int(clsID)], conf))
                cv2.rectangle(
                    image,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    image,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

            # Display the resulting image in Streamlit
            st.image(image, channels="BGR", caption="Object Detection Result")

            # Print detected classes and their confidence
            st.subheader("Detected Classes and Confidences:")
            for cls, conf in detected_classes:
                st.write(f"{cls}: {round(conf * 100, 2)}%")
else:
    st.warning('Please upload an image.')

