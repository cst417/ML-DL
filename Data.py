import numpy as np 
import matplotlib.pyplot as plt 
import os           #for directories
import cv2          #image operations
import random
import pickle
import tensorflow as tf 
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


DATADIR = "C:/Users/Chetan Tuli/Desktop/PetImages"
CATEGORIES = ["Dog", "Cat"]
#iterating through dogs and cats

img_size = 150
training_data = []

def create_training_data():
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)  #paths to cats or dogs dir 
        class_num = CATEGORIES.index(categories)    #assigning number for cats and dogs to differentiate
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  #converting images to array using cv2.imread and then reading them using os.path.join and then converting it to grayscale
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])  #appending data with training data variable
            except Exception as e:
                pass

create_training_data()
print(len(training_data))
random.shuffle(training_data)   #shuffle the data since everything first is all dogs and then all cats so neural network wont learn properly

#for sample in training_data[:10]:
    # print(sample[1])        #to check if labels are correct. 

X = []  #packing shuffle data into variable X and Y where X is feature set and Y is labels 
Y = []  

for features,label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)  #1 because it's a grayscale 
pickle_out = open("X.pickel","wb")   #to save data
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickel","wb")   #to save data
pickle.dump(Y, pickle_out)
pickle_out.close()

# pickle_in = open("X.pickle", "rb")  #to load the saved data
# X = pickle.load(pickle_in) 