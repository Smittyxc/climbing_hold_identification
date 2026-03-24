from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 1. Load your locally trained YOLO model into SAHI
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path=r'C:\Users\matts\winter2026\cis378\proj\best3.pt',
    confidence_threshold=0.50,
    device="cpu" 
)

image_path = r'C:\Users\matts\winter2026\cis378\proj\newwall3.webp'

# 2. Run the sliced inference!
# We are slicing the image into 512x512 squares with a 20% overlap 
# so holds on the edges of the slices don't get cut in half.
result = get_sliced_prediction(
    image_path,
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)


clean_predictions = []
MIN_PIXEL_AREA = 400  # Adjust this: 300 pixels is roughly a 17x17 pixel square

# Loop through everything SAHI found
for obj in result.object_prediction_list:
    # SAHI provides the exact area of the bounding box
    box_area = obj.bbox.area 
    
    if box_area > MIN_PIXEL_AREA:
        clean_predictions.append(obj)
    else:
        print(f"Filtered out tiny bolt hole (Area: {box_area})")

# 4. Overwrite SAHI's raw list with our clean, filtered list
result.object_prediction_list = clean_predictions


# 3. Export the visual result to see how it did
result.export_visuals(export_dir=r'C:\Users\matts\winter2026\cis378\proj', file_name="sahi_result", hide_conf=True, hide_labels=True)
print("Done! Check your folder for the sahi_result.png file.")