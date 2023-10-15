# Install libraries

import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization, Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D, Activation, Dense, Flatten, Input, add
from keras.models import Model
from keras import regularizers, backend as K
from sklearn.metrics import classification_report

# Read data csv
data = pd.read_csv("data.csv")
images = data["image"]
print(images[1].shape)

# Resize from (28, 28) to (32, 32)
images = [cv2.resize(img, (32, 32)) for img in data["image"]]
images = np.array(images, dtype = "float32")

print(images.shape)

# Preprocessing
images = np.expand_dims(images, axis = -1) # Add channel dimension
images /= 255.0 # Keep values between 0 and 1

print(images.shape)

# Convert labels from integer to vector for easier model fitting
lb = LabelBinarizer()
labels = lb.fit_transform(data["label"])

# Weights for each character
class_totals = labels.sum(axis = 0)

class_weight = {}

# Loop over classes and calculate class weights
for i in range(0, len(class_totals)):
  class_weight[i] = class_totals.max() / class_totals[i]

# Split data into train and test values
x_train, x_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size = 0.25,
    stratify = labels,
    random_state = 30,
)

# Data augmentation to improve results
data_augment = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.05,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.15,
    horizontal_flip = False,
    fill_mode = "nearest"
)

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, channel_dim, reduce = False, reg = 0.0001, eps = 2e-5, mom = 0.9):
        # Shortcut: initialize as input data
        shortcut = data

        # First block of ResNet module is 1x1 CONVs
        batch1 = BatchNormalization(axis = channel_dim, epsilon = eps, momentum = mom)(data)
        activation1 = Activation("relu")(batch1)
        convolution1 = Conv2D(int(K * 0.25), (1, 1), use_bias = False, kernel_regularizer = regularizers.L2(reg))(activation1)

        # Second block of ResNet module is 3x3 CONVs
        batch2 = BatchNormalization(axis = channel_dim, epsilon = eps, momentum = mom)(convolution1)
        activation2 = Activation("relu")(batch2)
        convolution2 = Conv2D(int(K * 0.25), (3, 3), strides = stride, padding = "same", use_bias = False, kernel_regularizer = regularizers.L2(reg))(activation2)

        # Third block of ResNet module is 1x1 CONVs
        batch3 = BatchNormalization(axis = channel_dim, epsilon = eps, momentum = mom)(convolution2)
        activation3 = Activation("relu")(batch3)
        convolution3 = Conv2D(K, (1, 1), use_bias = False, kernel_regularizer = regularizers.L2(reg))(activation3)

        if reduce:
            shortcut = Conv2D(K, (1, 1), strides = stride, use_bias = False, kernel_regularizer = regularizers.L2(reg))(activation1)

        # Add shortcut and final CONV
        final = add([convolution3, shortcut])

        return final

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg = 0.0001, eps = 2e-5, mom = 0.9):
        # Initialize input with channel last and channel dimensions
        input_shape = (height, width, depth)
        channel_dim = -1

        # Channel first, update shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channel_dim = 1

        # Set input and apply BatchNormalization
        inputs = Input(shape = input_shape)
        result = BatchNormalization(axis = channel_dim, epsilon = eps, momentum = mom)(inputs)

        #result = Conv2D(filters[0], (3, 3), use_bias = False, padding = "same", kernel_regularizer = regularizers.L2(reg))(result)

        for i in range(0, len(stages)):
            # Initialize stride and apply residual module to reduce spatial size of input volume
            stride = (1, 1) if i == 0 else (2, 2)
            result = ResNet.residual_module(result, filters[i + 1], stride, channel_dim, reduce = True, eps = eps, mom = mom)

            # Loop through layers in stage
            for j in range(0, stages[i] - 1):
                # ResNet module
                result = ResNet.residual_module(result, filters[i + 1], (1, 1), channel_dim, eps = eps, mom = mom)

        # Apply BN, ACT, POOL
        result = BatchNormalization(axis = channel_dim, epsilon = eps, momentum = mom)(result)
        result = Activation("relu")(result)
        result = AveragePooling2D((8, 8))(result)

        # Softmax classifier
        result = Flatten()(result)
        result = Dense(classes, kernel_regularizer = regularizers.L2(reg))(result)
        result = Activation("softmax")(result)

        # Create model
        model = Model(inputs, result, name = "resnet")

        return model
    
EPOCHS = 50
LEARNING_RATE = 1e-1
BS = 128

# Stochastic gradient descent optimizer
opt = tf.keras.optimizers.SGD(learning_rate = LEARNING_RATE, weight_decay = LEARNING_RATE / BS)

model = ResNet.build(32, 32, 1, len(lb.classes_), (3, 3, 3), (64, 64, 128, 256), reg = 0.0005)

model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

history = model.fit(
    data_augment.flow(x_train, y_train, batch_size = BS),
    validation_data = (x_test, y_test),
    steps_per_epoch = len(x_train) // BS,
    epochs = EPOCHS,
    class_weight = class_weight,
    verbose = 1
)

# Evaluate for 50 epochs
class_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
class_names = [i for i in class_names]
predictions = model.predict(x_test, batch_size = BS)
print(classification_report(y_test.argmax(axis = 1), 
                            predictions.argmax(axis = 1), 
                            target_names = class_names, 
                            zero_division = 0))

EPOCHS = 30

history = model.fit(
    data_augment.flow(x_train, y_train, batch_size = BS),
    validation_data = (x_test, y_test),
    steps_per_epoch = len(x_train) // BS,
    epochs = EPOCHS,
    class_weight = class_weight,
    verbose = 1
)

# Evaluate for 80 epochs
class_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
class_names = [i for i in class_names]
predictions = model.predict(x_test, batch_size = BS)
print(classification_report(y_test.argmax(axis = 1), 
                            predictions.argmax(axis = 1), 
                            target_names = class_names, 
                            zero_division = 0))

EPOCHS = 20

history = model.fit(
    data_augment.flow(x_train, y_train, batch_size = BS),
    validation_data = (x_test, y_test),
    steps_per_epoch = len(x_train) // BS,
    epochs = EPOCHS,
    class_weight = class_weight,
    verbose = 1
)

# Evaluate for 100 epochs
class_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
class_names = [i for i in class_names]
predictions = model.predict(x_test, batch_size = BS)
print(classification_report(y_test.argmax(axis = 1), 
                            predictions.argmax(axis = 1), 
                            target_names = class_names, 
                            zero_division = 0))

history = model.fit(
    data_augment.flow(x_train, y_train, batch_size = BS),
    validation_data = (x_test, y_test),
    steps_per_epoch = len(x_train) // BS,
    epochs = EPOCHS,
    class_weight = class_weight,
    verbose = 1
)

# Evaluate for 120 epochs
class_names = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
class_names = [i for i in class_names]
predictions = model.predict(x_test, batch_size = BS)
print(classification_report(y_test.argmax(axis = 1), 
                            predictions.argmax(axis = 1), 
                            target_names = class_names, 
                            zero_division = 0))

model.save_weights("model.h5")