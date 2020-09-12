#importing the necessary library
import math
import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf

#defining the shape of the model
height = 64
width = 64
depth = 3

#training the model
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(38))
model.add(Activation("softmax"))
model.summary()

INIT_LR = 1e-3
EPOCHS = 10
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# distribution
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
print("[INFO] training network...")

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = tf.keras.preprocessing.image.DirectoryIterator(
    'dataset/dataset/train', train_datagen, target_size=(64, 64), color_mode='rgb',
    classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None,
    data_format=None, save_to_dir=None, save_prefix='', save_format='png',
    follow_links=False, subset=None, interpolation='nearest', dtype=None
)

test_set = tf.keras.preprocessing.image.DirectoryIterator(
        'dataset/dataset/valid',test_datagen,target_size=(64,64),color_mode ='rgb',
        class_mode ='categorical',batch_size=32,shuffle=True,save_format='png',interpolation=
        'nearest')


batch_size=32

training_size = 70295

validation_size = 17572

# We take the ceiling because we do not drop the remainder of the batch
compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))

steps_per_epoch = compute_steps_per_epoch(training_size)
val_steps = compute_steps_per_epoch(validation_size)

model.fit_generator(training_set,
                         steps_per_epoch = steps_per_epoch,
                         epochs = EPOCHS,
                         validation_data = test_set,
                         validation_steps = val_steps)


print("[INFO] Calculating model accuracy")
scores = model.evaluate(test_set)
print(f"Test Accuracy: {scores[1]*100}")

#saving the model
print("[INFO]Saving the model....")
model.save("plant_disease_model.h5")

#testing new images
def converted_image(path):
    test_image = image.load_img(path,target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = test_image.reshape(1,64,64,3)
    test_image = test_image /255.
    return test_image
def dictionary_key(dictionary,key_value):
    list_of_key = list()
    list_of_values = dictionary.items()
    for items in list_of_values:
        if items[1]== key_value:
            list_of_key.append(items[0])
    return list_of_key
def prediction(img):
    pred = model.predict(img)[0]
    max_posibility = np.argmax(pred)
    key_name = dictionary_key(training_set.class_indices,max_posibility)
    for names in key_name:
        print("The Given Image belongs to :",names)
        
test_image = converted_image('dataset/test/test/hello.JPG')
prediction(test_image)
