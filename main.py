import os
from datetime import datetime
from yolo_inference import yolo_inference
from sahi_inference import sahi_detection_model

test_image_path = r".\test_images"
all_entries = os.listdir(test_image_path)
test_files = [os.path.join(test_image_path, f) for f in all_entries if os.path.isfile(os.path.join(test_image_path, f))]

now = datetime.now()
now_str = now.strftime("%m-%d-%Y_%H-%M-%S")
temp_path = "results/" + now_str 

try:
    os.makedirs(temp_path + "/sahi", exist_ok=True)
    os.makedirs(temp_path + "/yolo", exist_ok=True)
    print("Created directory: ./" + temp_path + "/")
except OSError as error:
    print("Error! Unable to create directory: ./" + temp_path + "/")

# --- Define CSV output paths ---
sahi_csv_file = f"{temp_path}/sahi_metrics.csv"
yolo_csv_file = f"{temp_path}/yolo_metrics.csv"

# --- Run the models! ---
print("\n--- Starting SAHI Inference ---")
# sahi_detection_model(temp_path + '/sahi', test_files, sahi_csv_file)

print("\n--- Starting YOLO Inference ---")
yolo_inference(test_files, temp_path + '/yolo', yolo_csv_file)
