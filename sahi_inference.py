import csv
import os
import time
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from datetime import datetime
import json
from pprint import pprint

def sahi_detection_model(save_path, test_paths, csv_save_path, conf=0.50):
    MIN_PIXEL_AREA = 400  
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=r'best4.pt', 
        confidence_threshold=conf,
        device="cpu" 
    )

    with open(csv_save_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image_Name", "Holds_Found", "Runtime_Seconds", "Avg_Confidence"])

        for i, img_path in enumerate(test_paths):
            start_time_iteration = time.time()
            width, height = get_image_size(img_path)

            result = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=height // 2,
                slice_width=width // 2,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
            
            result.object_prediction_list = filter_large_box(width, height, result.object_prediction_list)
            result.export_visuals(export_dir=save_path, file_name=f"result{i}", hide_conf=True, hide_labels=True)
            
            end_time_iteration = time.time()
            iteration_time = end_time_iteration - start_time_iteration
            
            holds_found = len(result.object_prediction_list)
            
            if holds_found > 0:
                avg_conf = sum([pred.score.value for pred in result.object_prediction_list]) / holds_found
            else:
                avg_conf = 0.0

            file_name = os.path.basename(img_path)
            writer.writerow([file_name, holds_found, round(iteration_time, 4), round(avg_conf, 4)])
            print(f"SAHI: {file_name} processed in {iteration_time:.2f} seconds.")

def get_image_size(path: str):
    with Image.open(path) as img:
        return img.size

def filter_large_box(width, height, preds): 
    filtered_preds = []
    img_size = width * height
    for pred in preds:
        box_area = pred.bbox.area
        if box_area < img_size / 25:
            filtered_preds.append(pred)
    return filtered_preds

def sahi_single_inf(img_path, save_path):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=r'best4.pt', 
        confidence_threshold=0.5,
        device="cpu" 
    )
    
    width, height = get_image_size(img_path)

    result = get_sliced_prediction(
                img_path,
                detection_model,
                slice_height=height // 2,
                slice_width=width // 2,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
            )
    
    result.object_prediction_list = filter_large_box(width, height, result.object_prediction_list)
    pprint(result.object_prediction_list)
    app_data = []
    for box_data in result.object_prediction_list:
        
        bbox = {
            'coords': box_data.bbox.to_xywh()
        }

    app_data = {
        'date': datetime.isoformat(datetime.now()),
        'boxes': result.object_prediction_list,
    }
   
    with open("app_data.json", "w") as f:
        json.dump(app_data, f)
