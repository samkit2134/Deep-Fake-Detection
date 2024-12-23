import json
import os
from distutils.dir_util import copy_tree
import shutil
import numpy as np
import splitfolders

base_path = '.\\train_sample_videos\\'
dataset_path = '.\\prepared_dataset\\'
print('Creating Directory: ' + dataset_path)
os.makedirs(dataset_path, exist_ok=True)

tmp_fake_path = '.\\tmp_fake_faces'
print('Creating Directory: ' + tmp_fake_path)
os.makedirs(tmp_fake_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

real_path = os.path.join(dataset_path, 'real')
print('Creating Directory: ' + real_path)
os.makedirs(real_path, exist_ok=True)

fake_path = os.path.join(dataset_path, 'fake')
print('Creating Directory: ' + fake_path)
os.makedirs(fake_path, exist_ok=True)

for filename in metadata.keys():
    print(filename)
    print(metadata[filename]['label'])
    tmp_path = os.path.join(os.path.join(base_path, get_filename_only(filename)), 'faces')
    print(tmp_path)
    if os.path.exists(tmp_path):
        if metadata[filename]['label'] == 'REAL':    
            print('Copying to :' + real_path)
            copy_tree(tmp_path, real_path)
        elif metadata[filename]['label'] == 'FAKE':
            print('Copying to :' + tmp_fake_path)
            copy_tree(tmp_path, tmp_fake_path)
        else:
            print('Ignored..')

all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
print('Total Number of Real faces: ', len(all_real_faces))

all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]
print('Total Number of Fake faces: ', len(all_fake_faces))

random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)
for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

print('Down-sampling Done!')

# Split into Train/ Val/ Test folders
splitfolders.ratio(dataset_path, output='split_dataset', seed=1377, ratio=(.8, .1, .1)) # default values
print('Train/ Val/ Test Split Done!')



# 02-prepare_fake_real_dataset.py
# Purpose: This script organizes the cropped face images into a structured dataset and splits it into training, validation, and test sets.
# Usage:
# Copies the cropped face images into a new directory structure.
# Down-samples the dataset if necessary.
# Uses the splitfolders library to split the dataset into training, validation, and test sets.
# Example code excerpt:

# Import Necessary Libraries:

# json: For handling JSON data.
# os: For interacting with the operating system.
# distutils.dir_util: For copying directories.
# shutil: For high-level file operations.
# numpy: For numerical operations.
# splitfolders: For splitting the dataset into training, validation, and test sets.
# Define Paths:

# base_path: The base directory containing the original video frames and metadata.
# dataset_path: The directory where the prepared dataset will be stored.
# tmp_fake_path: A temporary directory for storing fake faces.
# Function to Get Filename Only:

# This function extracts the filename without the extension from a given file path.
# Load Metadata:

# Loads the metadata from metadata.json, which contains labels indicating whether a video is real or fake.
# Organize Images:

# Iterates over the metadata and copies the cropped face images to the appropriate directories (real or tmp_fake_faces) based on their labels.
# Down-sample Fake Images:

# Ensures that the number of fake images matches the number of real images by randomly selecting a subset of fake images.
# Split the Dataset:

# Uses the splitfolders library to split the dataset into training, validation, and test sets based on the specified ratio (80% training, 10% validation, 10% test).
# Summary
# The prepared_dataset folder is an intermediate step in the data preparation process. It organizes the cropped face images into separate directories for real and fake images. This organization makes it easier to split the dataset into training, validation, and test sets later. The 02-prepare_fake_real_dataset.py script is responsible for creating this folder and populating it with the appropriate images. This structured approach ensures that the dataset is well-organized and ready for the next steps in the machine learning pipeline.