from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image
import time

data = {
    'img_name': ['result0', 'result1', 'result2', 'result' ],
    'true_hold_count': [],
    'est_hold_count': []
}
def sahi_detection_model(save_path, test_paths, conf=0.50):
    MIN_PIXEL_AREA = 400  
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=r'C:\Users\matts\winter2026\cis378\climbing_hold_identification\best4.pt',
        confidence_threshold=conf,
        device="cpu" 
    )

    for i, img_path in enumerate(test_paths):
        start_time_iteration = time.time()
        width, height = get_image_size(img_path)

        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=height // 2,
            slice_width=width // 2,
            overlap_height_ratio=0.4,
            overlap_width_ratio=0.4,
        )
        
        result.object_prediction_list = filter_large_box(width, height, result.object_prediction_list)
        result.export_visuals(export_dir=save_path, file_name=f"result{i}", hide_conf=True, hide_labels=True)
        
        end_time_iteration = time.time()
        iteration_time = end_time_iteration - start_time_iteration
        print(f"Iteration {i+1} took: {iteration_time} seconds")

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

