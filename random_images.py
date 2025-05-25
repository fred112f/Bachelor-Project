import os
import glob
import random
import shutil

SEED = 42
TRAIN_SPLIT = 0.8  
datasetA_dir = "datasets/screenshots_Open" 
datasetB_dir = "datasets/screenshots_SE" 
output_base_dir = "split_dataset"
input_images_paths  = sorted(glob.glob(os.path.join(datasetA_dir, "*.png")))
target_images_paths = sorted(glob.glob(os.path.join(datasetB_dir, "*.png")))

assert len(input_images_paths) == len(target_images_paths), \
    "Dataset A and B must have the same number of images."
num_images = len(input_images_paths)
random.seed(SEED) 
indices = list(range(num_images))
random.shuffle(indices)
train_size = int(TRAIN_SPLIT * num_images)
train_indices = indices[:train_size]
val_indices   = indices[train_size:]
train_input_paths  = [input_images_paths[i] for i in train_indices]
train_target_paths = [target_images_paths[i] for i in train_indices]
val_input_paths  = [input_images_paths[i] for i in val_indices]
val_target_paths = [target_images_paths[i] for i in val_indices]
train_a_dir = os.path.join(output_base_dir, "train", "A")
train_b_dir = os.path.join(output_base_dir, "train", "B")
val_a_dir   = os.path.join(output_base_dir, "val", "A")
val_b_dir   = os.path.join(output_base_dir, "val", "B")
os.makedirs(train_a_dir, exist_ok=True)
os.makedirs(train_b_dir, exist_ok=True)
os.makedirs(val_a_dir, exist_ok=True)
os.makedirs(val_b_dir, exist_ok=True)

for path in train_input_paths:
    shutil.copy(path, train_a_dir)
for path in train_target_paths:
    shutil.copy(path, train_b_dir)
for path in val_input_paths:
    shutil.copy(path, val_a_dir)
for path in val_target_paths:
    shutil.copy(path, val_b_dir)

print("Data split and copied successfully!")
