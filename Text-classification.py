#reference from https://www.tensorflow.org/tutorials/keras/basic_text_classification
#https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
import tensorflow as tf 
import numpy as np 

imdb = tf.keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)  #The argument num_words=10000 keeps the top 10,000 most frequently occurring words in the training data. The rare words are discarded to keep the size of the data manageable.

print("Training Entries: {}, labels:{}".format(len(train_data), len(train_labels)))

# print(train_data[0])

print(len(train_data[0]), len(train_data[1]))      #Shows the number of words in the first and second reviews

#Convert the integers back to words

# A dictionary mapping words to an integer index. 
word_index = imdb.get_word_index()

# Introducing new keywords.Reserved keywords
word_index = {k:(v+3) for k,v in word_index.items()}            #k = key ,  v= value
word_index["<PAD>"] = 0             #padding
word_index["<START>"] = 1           #start
word_index["<UNK>"] = 2             #unknown - when you don't know what the word is
word_index["<UNUSED>"] = 3          #unused - when words are unused 

#flips the integer index to word
reverse_word_index = dict([[value, key] for (key, value) in word_index.items()])    #reversing the mapping. Flipping from value,key to key value

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])         #going through each of the entries in the text list which will list integers and checks if the integer is present in the reverse word index. If present then the integer is replaced with the word. Eg 105 might correspond to a word. If the word is not defined, then it would return a ?.


# Prepare the data
# The reviews—the arrays of integers—must be converted to tensors before fed into the neural network. This conversion can be done a couple of ways:

# Convert the arrays into vectors of 0s and 1s indicating word occurrence, similar to a one-hot encoding. For example, the sequence [3, 5] would become a 10,000-dimensional vector that is all zeros except for indices 3 and 5, which are ones. Then, make this the first layer in our network—a Dense layer—that can handle floating point vector data. This approach is memory intensive, though, requiring a num_words * num_reviews size matrix.

# Alternatively, we can pad the arrays so they all have the same length, then create an integer tensor of shape max_length * num_reviews. We can use an embedding layer capable of handling this shape as the first layer in our network.
train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],padding='post',maxlen=256)
test_data = tf.keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"],padding='post',maxlen=256)

# print(len(train_data[0]), len(test_data[0]))

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 16))        #The first layer is an Embedding layer. 
                                                            #This layer takes the integer-encoded vocabulary and looks up the 
                                                            # embedding vector for each word-index. These vectors are learned 
                                                            # as the model trains. The vectors add a dimension to the output 
                                                            # array. The resulting dimensions are: (batch, sequence, embedding)
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))         #This fixed-length output vector is piped through a 
#                                                                   fully-connected (Dense) layer with 16 hidden units.
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))       #The last layer is densely connected with a single output node. 
                                                                    #Using the sigmoid activation function, this value is a float 
                                                                    # between 0 and 1, representing a probability, or confidence level.
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

#When training, we want to check the accuracy of the model on data it hasn't seen before. 
#Create a validation set by setting apart 10,000 examples from the original training data.
# (Why not use the testing set now? Our goal is to develop and tune our model using only the training data, 
# then use the test data just once to evaluate our accuracy).

x_val = train_data[:10000]
partial_x_train = train_data[:10000]

y_val = train_labels[:10000]
partial_y_train = train_labels[:10000]

#training the model
#Train the model for 40 epochs in mini-batches of 512 samples. This is 40 iterations over all samples in the x_train and y_train tensors. 
# While training, monitor the model's loss and accuracy on the 10,000 samples from the validation set

history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=40, validation_data=(x_val,y_val), verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)










#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.