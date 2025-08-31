
#so in this file we will be building a neural network to classify images of clothing from the fashion mnist dataset



import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # names of the classes

plt.figure()
plt.imshow(train_images[8000])
plt.colorbar()
plt.grid(False)
plt.show()

# so lets do data preprocessing which is important in deep learning
train_images = train_images / 255.0
test_images = test_images / 255.0
# we are scaling the values between 0 and 1 by dividing it by 255.0

#lets build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

# compiling the model
model.compile(optimizer='adam', # optimizer is used to update the weights based on the loss function
              loss='sparse_categorical_crossentropy',  # these are hyperparameters and we can change them 
              metrics=['accuracy'])
# sparse_categorical_crossentropy is used when there are more than 2 classes
# categorical_crossentropy is used when there are only 2 classes

model.fit(train_images, train_labels, epochs=1)  # we pass the data, labels and epochs and watch the magic!

# evaluating accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) # verbose is used to show the progress bar
print('Test accuracy:', test_acc)


# making predictions
predictions = model.predict(test_images) # this will return a list of 10 numbers for each image











# lets create a function to predict the image
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 60000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
print("The expected label is " + class_names[label])
print("The predicted label is " + class_names[np.argmax(predictions[num])])


























