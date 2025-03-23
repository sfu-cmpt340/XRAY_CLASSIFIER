import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import hashlib

# Paths and categories
dataset_path = "dataset_used(combination_of_three)/"
categories = ["COVID", "PNEUMONIA", "NORMAL", "TUBERCULOSIS"]

# Output folder
output_path = "prepared_dataset/"
splits = ["train", "val", "test"]

# Create output directories for train/val/test
for split in splits:
    for category in categories:
        os.makedirs(os.path.join(output_path, split, category), exist_ok=True)

# Create output directory for blurry images for inspection
blurry_output_dir = os.path.join(output_path, "blurry")
os.makedirs(blurry_output_dir, exist_ok=True)

# Hash function for duplicates
def calculate_hash(img_path):
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

# Blurry check function with adjusted threshold
def is_blurry(image, threshold=59):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

# Low contrast check
def is_low_contrast(image, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std() < threshold

# Function to save blurry images for visual inspection
def save_blurry_images(blurry_paths, save_dir):
    for img_path in blurry_paths:
        try:
            img = cv2.imread(img_path)
            if img is not None:
                file_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(save_dir, file_name), img)
        except Exception as e:
            print(f"Error saving blurry image {img_path}: {e}")

image_hashes = set()
duplicates, blurry_imgs, low_contrast_imgs = [], [], []

image_data, labels = [], []

print("Starting processing with automated quality checks...")

# Process images
for category in categories:
    path = os.path.join(dataset_path, category)
    img_names = os.listdir(path)
    img_paths = [os.path.join(path, img_name) for img_name in img_names]

    for img_path in img_paths:
        img_hash = calculate_hash(img_path)
        if img_hash in image_hashes:
            duplicates.append(img_path)
            continue

        image_hashes.add(img_hash)

        try:
            img = cv2.imread(img_path)
            if is_blurry(img):
                blurry_imgs.append(img_path)
                continue
            if is_low_contrast(img):
                low_contrast_imgs.append(img_path)
                continue

            img = cv2.resize(img, (224, 224))
            # Normalize and convert to float32 to save memory
            img = (img / 255.0).astype(np.float32)
            image_data.append(img)
            labels.append(category)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print(f"\nDuplicates removed: {len(duplicates)}")
print(f"Blurry images removed: {len(blurry_imgs)}")
print(f"Low-contrast images removed: {len(low_contrast_imgs)}")

# Save blurry images for visual inspection
save_blurry_images(blurry_imgs, blurry_output_dir)
print(f"Saved blurry images for inspection in: {blurry_output_dir}")

# Convert to arrays with float32 to reduce memory usage
image_data = np.array(image_data, dtype=np.float32)
labels = np.array(labels)

# Encode labels
label_encoder = {label: idx for idx, label in enumerate(categories)}
encoded_labels = np.array([label_encoder[label] for label in labels])

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(
    image_data, labels, test_size=0.15, random_state=42, stratify=labels)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# Save images to folders
def save_images_to_folder(images, labels, split_name):
    for img_array, label in zip(images, labels):
        save_dir = os.path.join(output_path, split_name, label)
        file_name = f"{hashlib.md5(img_array).hexdigest()}.png"
        img_save_path = os.path.join(save_dir, file_name)
        # Convert back to 8-bit for saving
        img_bgr = (img_array * 255).astype(np.uint8)
        cv2.imwrite(img_save_path, cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR))

print("Saving images into category folders... (this may take a while)")

save_images_to_folder(X_train, y_train, "train")
save_images_to_folder(X_val, y_val, "val")
save_images_to_folder(X_test, y_test, "test")

# Also save numpy arrays
np.save(os.path.join(output_path, "X_train.npy"), X_train)
np.save(os.path.join(output_path, "X_val.npy"), X_val)
np.save(os.path.join(output_path, "X_test.npy"), X_test)
np.save(os.path.join(output_path, "y_train.npy"), [label_encoder[y] for y in y_train])
np.save(os.path.join(output_path, "y_val.npy"), [label_encoder[y] for y in y_val])
np.save(os.path.join(output_path, "y_test.npy"), [label_encoder[y] for y in y_test])

print("\n✅ Data fully prepared and saved into both category folders and numpy arrays!")
