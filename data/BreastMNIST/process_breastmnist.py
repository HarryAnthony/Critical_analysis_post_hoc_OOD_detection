
'''
Code for processing the BreastMNIST dataset
'''
import os
import numpy as np
import csv
import pandas as pd
from PIL import Image

# Base directory to save outputs
breastmnist_save_dir = ''

# Define paths to .npy files
data_files = {
    'train': {
        'images': 'train_images.npy',
        'labels': 'train_labels.npy',
    },
    'val': {
        'images': 'val_images.npy',
        'labels': 'val_labels.npy',
    },
    'test': {
        'images': 'test_images.npy',
        'labels': 'test_labels.npy',
    }
}

# Load the breastmnist_map.csv file
map_file_path = "breastmnist_map.csv"  # Update with the correct path
class_map = pd.read_csv(map_file_path, skiprows=1)

# Extract unique classes for one-hot encoding
classes = class_map['Class'].unique()
class_to_index = {class_name: i for i, class_name in enumerate(classes)}

# Create output base directory
if breastmnist_save_dir != '':
    os.makedirs(breastmnist_save_dir, exist_ok=True)

# Process each split
for split, paths in data_files.items():
    # Load images and labels
    images = np.load(paths['images'])  
    labels = np.load(paths['labels']) 

    # Create output directory for the split
    output_dir = os.path.join(breastmnist_save_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    # Path for the CSV file
    if split == 'val':
        csv_file_path = os.path.join(breastmnist_save_dir, 'valid.csv')
    else:
        csv_file_path = os.path.join(breastmnist_save_dir, f'{split}.csv')

    # Write the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        # Define the column headers
        fieldnames = ['image_name', 'Path'] + list(classes)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header
        writer.writeheader()

        # Process each image
        for i, image in enumerate(images):
            # Generate the image path
            image_file_name = f'image_{i}.jpg'
            image_path = os.path.join(output_dir, image_file_name)

            # Save the image to the output directory
            Image.fromarray(image).convert('L').save(image_path)  # Convert to grayscale if needed

            # Get the class for this image
            map_row = class_map[class_map['Image'] == f'{split}/{image_file_name}']
            if not map_row.empty:
                new_class = map_row.iloc[0]['Class']
            else:
                continue  # Skip if the image is not in the map

            # One-hot encode the class
            one_hot = [1 if new_class == class_name else 0 for class_name in classes]

            # Write the row for this image
            writer.writerow({
                'image_name': image_file_name,
                'Path': image_path,
                **dict(zip(classes, one_hot))
            })
