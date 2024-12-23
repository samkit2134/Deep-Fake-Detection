import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os


model_path = 'C:\Projects\deep_fake_detection\DeepFake-Detect\deepfake_detector_model.h5' #apan yaha load kr rhe h
model = load_model(model_path)


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64)) #cropping the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # here  we are doing Rescale the image
    return img_array


def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction[0][0]


def main():  #apna main function 
    
    images_dir = 'C:\Projects\deep_fake_detection\DeepFake-Detect\images_to_predict' # Directory containing images to predict
    
  
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist.")
        return

    
   
    for img_name in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_name)
        if os.path.isfile(img_path):
            prediction = predict_image(img_path)
            if prediction > 0.38:
                print(f'The image {img_name} is predicted to be a DEEPFAKE with a confidence of {prediction:.2f}')
            else:
                print(f'The image {img_name} is predicted to be REAL with a confidence of {1 - prediction:.2f}')
    

if __name__ == "__main__":
    main()
    

