import numpy as np
import json
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

import argparse
        
def process_image(image):
    image = tf.image.resize(image, [224,224])
    image /= 255
    return image.numpy()

def predict(image_path, model, class_names, top_k=5):
    image = Image.open(image_path)
    image_np_arr = np.asarray(image)
    processed_image = process_image(image_np_arr)
    image = np.expand_dims(processed_image,axis=0)
    prediction = model.predict(image)
    indices_top_k = prediction.argsort()[0][-top_k:][::-1]
    classes = [class_names[str(i+1)] for i in indices_top_k] 
    probs = [prediction[0][i] for i in indices_top_k] 
    return probs, classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument("image_path",help="Image Path", default="")
    parser.add_argument("saved_model",help="Model Path", default="")
    parser.add_argument("--top_k", help="Top k predictions", required = False, default = 5)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        
    saved_model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
    print(saved_model.summary())
    probs, classes = predict(args.image_path, saved_model, class_names, int(args.top_k))
    print('\tClasses Probilities\n'
    print('\n'.join([str(classes[i] +'\t'+ str(probs[i])) for i in range(int(args.top_k))]))
