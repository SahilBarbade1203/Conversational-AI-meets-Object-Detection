from roboflow import Roboflow
rf = Roboflow(api_key="Gb6Bdsa1RAucMq5EsDTY")
project = rf.workspace("iconsside2").project("modelchoose")
version = project.version(1)
dataset = version.download("yolov8")

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolov8n.yaml").load("yolov8n.pt")

# Train the model
results = model.train(data=f"{dataset.location}/data.yaml", epochs=30, imgsz=640)

#training results
from IPython.display import Image

print(Image(filename = "/content/runs/detect/train2/confusion_matrix_normalized.png", width = 1000))
print(Image(filename = f"/content/runs/detect/train2/results.png", width = 1000))

#validating model by using best model's weight
# Load a model
weight_path = "/content/drive/MyDrive/SOC_2024/best.pt"
model = YOLO("yolov8n.pt")
model = YOLO(weight_path)  # load a custom model

# Validate the model
metrics = model.val()
print(metrics)

# Run inference on test images with arguments
model.predict("/content/modelChoose-1/test/images", save=True, imgsz=320, conf=0.5)

#manually visualizing each detected engine failure images
import os
from IPython.display import Image, display

directory = "/content/runs/detect/predict"
for img_path in os.listdir(directory):
  full_path = os.path.join(directory, img_path)
  display(Image(filename = full_path, width = 320))
