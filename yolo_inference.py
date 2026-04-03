from ultralytics import YOLO
import time
import csv
import os

def yolo_inference(test_paths, save_path, csv_save_path):      
    model = YOLO('best4.pt') 

    with open(csv_save_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Image_Name", "Holds_Found", "Runtime_Seconds", "Avg_Confidence"])

        for i, img_path in enumerate(test_paths):
            start_time_iteration = time.time()

            # Predict on the single image
            results = model.predict(source=img_path, conf=0.3, iou=0.2, imgsz=1280, show=False, verbose=False)
            result = results[0] # Extract the single Result object

            # Filter out large boxes
            valid_indices = filter_yolo_large_box(result.orig_shape, result.boxes)
            filtered_result = result[valid_indices]
            
            # Save the image
            filtered_result.save(filename=f"{save_path}/yolo_result_{i}.jpg", labels=False)

            # 3. Calculate runtime
            end_time_iteration = time.time()
            iteration_time = end_time_iteration - start_time_iteration
            
            # 4. Calculate metrics
            holds_found = len(result.boxes)
            
            # Calculate average confidence from the tensor
            if holds_found > 0:
                avg_conf = result.boxes.conf.mean().item()
            else:
                avg_conf = 0.0

            # 5. Write the data row to the CSV
            file_name = os.path.basename(img_path)
            writer.writerow([file_name, holds_found, round(iteration_time, 4), round(avg_conf, 4)])
            print(f"YOLO: {file_name} processed in {iteration_time:.2f} seconds.")

def filter_yolo_large_box(orig_shape, boxes):
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