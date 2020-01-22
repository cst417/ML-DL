import cv2 
import tensorflow as tf 

categories = ["Dog", "Cat"]
path = 'C:/Users/Chetan Tuli/Desktop/Samples/cat.png'

def prepare(filepath):
    img_size = 150
    try:
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (img_size, img_size))
        return new_array.reshape(-1, img_size, img_size, 1)
    except Exception as e:
        print(str(e))

model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare(path)])

print(categories[int(prediction)])
