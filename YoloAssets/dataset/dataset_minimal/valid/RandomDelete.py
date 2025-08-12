import os
import random

# CONFIG
DeleteAmount = 3800  # number of images to delete
image_folder = "Images"
label_folder = "Labels"

# Get all images (we assume they are .jpg — change if needed)
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

# Ensure we don't try to delete more than available
DeleteAmount = min(DeleteAmount, len(image_files))

# Pick random files to delete
files_to_delete = random.sample(image_files, DeleteAmount)

for image_file in files_to_delete:
    # Paths
    image_path = os.path.join(image_folder, image_file)
    label_name = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(label_folder, label_name)

    # Delete image
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Deleted image: {image_file}")
    else:
        print(f"Image not found: {image_file}")

    # Delete label
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"Deleted label: {label_name}")
    else:
        print(f"Label not found: {label_name}")

print(f"\nDeleted {DeleteAmount} image/label pairs.")
