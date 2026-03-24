from ultralytics import YOLO
import json
from datetime import datetime
import os

now = datetime.now()
now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
temp_path = "results/" + now_str

try:
	os.makedirs(temp_path, exist_ok = True)
	print("Created directory: ./" + temp_path + "/")
except OSError as error:
	print("Error! Unable to create directory: ./" + temp_path + "/")
model = YOLO('best3.pt') 

image_path = './test_images'

results = model.predict(source=image_path, conf=0.25, iou=0.2, imgsz=1280, show=True, show_labels=False, show_conf=False)

for i, result in enumerate(results):
    result.save(filename=f"{temp_path}/result{i}.jpg", labels=False)

holds_database_records = []

print(results)
for box in results[0].boxes:
    
    # --- GET COORDINATES ---
    # xywh gets the Center X, Center Y, Width, and Height in pixels.
    # We use .tolist() to convert it from a PyTorch tensor to a standard Python list
    coords = box.xywh[0].tolist() 
    x_center, y_center, width, height = coords
    
    # --- GET CONFIDENCE ---
    # .item() extracts the single number from the tensor
    confidence = box.conf[0].item()
    
    # --- GET CLASS ID ---
    # Since you only have 1 class ("hold"), this will always be 0
    class_id = int(box.cls[0].item())
    
    # 4. Structure the data into a clean dictionary
    hold_data = {
        "class": "hold",
        "x": round(x_center, 2),
        "y": round(y_center, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "confidence": round(confidence, 2)
    }
    
    holds_database_records.append(hold_data)

json_output = json.dumps(holds_database_records, indent=4)
print(json_output)