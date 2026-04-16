# Ran in Google Collab T4
!pip install ultralytics roboflow
!pip install roboflow
from google.colab import drive
from roboflow import Roboflow
from ultralytics import YOLO

drive.mount('/content/drive')

# load dataset from Roboflow
rf = Roboflow(api_key="")
project = rf.workspace("matts-workspace-yqf3d").project("climbing-hold-detection-1r91o")
version = project.version(8)
dataset = version.download("yolo26")

model = YOLO('yolo26n.pt') 

drive_save_path = 'climbing_modelv8'

results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=1280,          
    batch=8,             
    device=0,
    patience=25,         
    project=drive_save_path, 
    name='nano_high_res'
)