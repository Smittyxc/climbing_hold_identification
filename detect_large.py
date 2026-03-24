import os

# 1. Point this to the folder containing your new YOLO .txt label files
labels_dir = r"C:\Users\matts\winter2026\cis378\proj\labels"

print("Scanning for massive bounding boxes...")

# 2. Loop through every text file
for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(labels_dir, filename), 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    # YOLO format: class x_center y_center width height
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # 3. Flag any box taking up more than 40% of the screen
                    if width > 0.20 or height > 0.20:
                        print(f"🚩 Giant box found in: {filename} (Width: {width:.2f}, Height: {height:.2f})")

print("Scan complete.")