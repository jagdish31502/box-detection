from ultralytics import YOLO

# Load a model
model = YOLO("weights\yolov8n.pt")  # build a new model from scratch

# Use the model
model.train(data="config.yaml", epochs=50)  # train the model