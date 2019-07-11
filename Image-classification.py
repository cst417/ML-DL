#https://www.youtube.com/watch?v=WvoLTXIjBYU&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=2
#https://www.youtube.com/watch?v=WvoLTXIjBYU&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=3

import tensorflow as tf 
import pickle 
import time

name = "Cats-Dogs-{}".format(int(time.time()))

tensorboard = tf.keras.callbacks.TensorBoard(log_dir = 'logs\{}'.format(name))        

x = pickle.load(open("X.pickel", "rb"))
y = pickle.load(open("Y.pickel", "rb"))

#normalize the data first
x = x/255.0

#making model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3,3), input_shape = x.shape[1:]))         #64 units, 3,3 window size
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64, (3,3)))            
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())            #converts the 3d features into 1d features
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Activation("relu"))

model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x, y, batch_size=32, validation_split=0.1, epochs=5, callbacks=[tensorboard])              #batch size = how many can be passed at once. You don't wanna pass 1 at a time or all at once



model.save('64x3-CNN.model')

