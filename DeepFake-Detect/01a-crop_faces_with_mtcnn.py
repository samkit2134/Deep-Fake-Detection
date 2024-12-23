
import cv2
from mtcnn import MTCNN
import sys
import os
import json
from keras import backend as K
import tensorflow as tf

print(tf.__version__)
tf.get_logger().setLevel('ERROR')

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_path = os.path.join('.', 'train_sample_videos')

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

for filename in metadata.keys():
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    print('Processing Directory: ' + tmp_path)
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
    faces_path = os.path.join(tmp_path, 'faces')
    print('Creating Directory: ' + faces_path)
    os.makedirs(faces_path, exist_ok=True)
    print('Cropping Faces from Images...')

    for frame in frame_images:
        print('Processing ', frame)
        detector = MTCNN()
        image_path = os.path.join(tmp_path, frame)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue
        result = detector.detect_faces(image)
        for i, face in enumerate(result):
            bounding_box = face['box']
            x, y, width, height = bounding_box
            cropped_face = image[y:y+height, x:x+width]
            face_filename = os.path.join(faces_path, f"{get_filename_only(frame)}_face_{i}.jpg")
            cv2.imwrite(face_filename, cropped_face)
            print(f"Saved cropped face to {face_filename}")
    
#01a-crop_faces_with_mtcnn.py
# Purpose: This script uses the MTCNN (Multi-task Cascaded Convolutional Networks) library to detect and crop faces from images.
# Usage:
# Loads images from the train_sample_videos directory.
# Detects faces in each image using MTCNN.
# Crops the detected faces and saves them in a subdirectory named faces within the same directory as the original images.


