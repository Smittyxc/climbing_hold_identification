from ultralytics import YOLO
import json
from datetime import datetime
import time
import os
from sahi_test import sahi_detection_model

test_image_path = r".\test_images"
all_entries = os.listdir(test_image_path)
test_files = [test_image_path + f"\\" +f for f in all_entries if os.path.isfile(os.path.join(test_image_path, f))]

now = datetime.now()
now_str = now.strftime("%m-%d-%Y_%H-%M-%S-%f")
temp_path = "results/" + now_str 

try:
	os.makedirs(temp_path + "/sahi", exist_ok = True)
	print("Created directory: ./" + temp_path + "/")
except OSError as error:
	print("Error! Unable to create directory: ./" + temp_path + "/")

def filter_large_box(orig_shape, boxes):
    img_height, img_width = orig_shape
    max_allowed_area = (img_height * img_width) / 25

    valid_indices = []
    for j, box in enumerate(boxes):
        w = box.xywh[0][2].item()
        h = box.xywh[0][3].item()
        box_area = w * h
        if box_area <= max_allowed_area:
            valid_indices.append(j)
    return valid_indices


def yolo_inference():      
    model = YOLO('best4.pt') 
    start_time_iteration = time.time()

    results = model.predict(source=test_image_path, conf=0.3, iou=0.2, imgsz=1280, show=True, show_labels=False, show_conf=False)

    for i, result in enumerate(results):
        valid_indices = filter_large_box(results[i].orig_shape, results[i].boxes)
        filtered_result = result[valid_indices]
        filtered_result.save(filename=f"{temp_path}/result{i}.jpg", labels=False)

    end_time_iteration = time.time()
    iteration_time = end_time_iteration - start_time_iteration
    print(f"{len(results)} took on average: {iteration_time / len(results)} seconds")

    # holds_database_records = []

    # for box in results[0].boxes:
        
    #     # --- GET COORDINATES ---
    #     # xywh gets the Center X, Center Y, Width, and Height in pixels.
    #     # We use .tolist() to convert it from a PyTorch tensor to a standard Python list
    #     coords = box.xywh[0].tolist() 
    #     x_center, y_center, width, height = coords
        
    #     # --- GET CONFIDENCE ---
    #     # .item() extracts the single number from the tensor
    #     confidence = box.conf[0].item()
        
    #     # --- GET CLASS ID ---
    #     # Since you only have 1 class ("hold"), this will always be 0
    #     class_id = int(box.cls[0].item())
        
    #     # 4. Structure the data into a clean dictionary
    #     hold_data = {
    #         "class": "hold",
    #         "x": round(x_center, 2),
    #         "y": round(y_center, 2),
    #         "width": round(width, 2),
    #         "height": round(height, 2),
    #         "confidence": round(confidence, 2)
    #     }
        
    #     holds_database_records.append(hold_data)

# json_output = json.dumps(holds_database_records, indent=4)
# print(json_output)
single_test = [f"./test_images/img_7225.jpeg"]
sahi_detection_model(temp_path + '/sahi', single_test)
# yolo_inference()