import os
import shutil
import random
from PIL import Image, ImageOps

# Paths
input_dir = r"D:\Research\5 Preprocessed\Output_Cropped2"
output_dir = r"D:\Research\5 Preprocessed\Leaf_Disease_Dataset6"

# Split ratios
train_ratio = 0.8
test_ratio = 0.1
val_ratio = 0.1

# Create output folder structure
for split in ["Train", "Test", "Validation"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

# Get all class folders
classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

# Find largest class size (target for augmentation)
max_count = max(len([f for f in os.listdir(os.path.join(input_dir, c)) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) for c in classes)
print(f"🎯 Target size per class after augmentation: {max_count} images")

def augment_image(image):
    """Apply a random augmentation to the image"""
    choice = random.choice(["flip", "rotate90", "rotate180", "rotate270"])
    if choice == "flip":
        return ImageOps.mirror(image)  # Horizontal flip
    elif choice == "rotate90":
        return image.rotate(90, expand=True)
    elif choice == "rotate180":
        return image.rotate(180, expand=True)
    elif choice == "rotate270":
        return image.rotate(270, expand=True)
    return image

for cls in classes:
    class_path = os.path.join(input_dir, cls)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    random.shuffle(images)

    # Augment until we reach max_count
    while len(images) < max_count:
        img_to_aug = random.choice(images)  # Pick a random image from the current list
        img_path = os.path.join(class_path, img_to_aug)
        
        with Image.open(img_path) as img:
            img = augment_image(img)
            aug_name = f"aug_{len(images)+1}.jpg"
            aug_path = os.path.join(class_path, aug_name)
            img.save(aug_path)
            images.append(aug_name)

    # Now all classes have max_count images
    for idx, img_name in enumerate(images, start=1):
        new_name = f"{cls}_{idx:03d}.jpg"
        img_path = os.path.join(class_path, img_name)

        # Decide split
        if idx <= int(train_ratio * max_count):
            split = "Train"
        elif idx <= int((train_ratio + test_ratio) * max_count):
            split = "Test"
        else:
            split = "Validation"

        split_class_path = os.path.join(output_dir, split, cls)
        os.makedirs(split_class_path, exist_ok=True)
        shutil.copy(img_path, os.path.join(split_class_path, new_name))

print("✅ Dataset augmented, balanced, renamed, and split successfully!")

