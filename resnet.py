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

# Download EMNIST data
from emnist import extract_training_samples, extract_test_samples
train_images, train_labels = extract_training_samples('balanced')
test_images, test_labels = extract_test_samples('balanced')

# Combine train images and test images (vstack for more than 3 dimensions)
images = np.vstack([train_images, test_images])

# Combine train labels and test labels (hstack for up to 3 dimensions)
labels = np.hstack([train_labels, test_labels])

print(images.shape)

# Resize from (28, 28) to (32, 32)
images = [cv2.resize(img, (32, 32)) for img in images]
images = np.array(images, dtype = "float32")

print(images.shape)

# Preprocessing
images = np.expand_dims(images, axis = -1)
images /= 255.0

print(images.shape)

# Convert labels from integer to vector for easier model fitting + count weights in characters and classes
le = LabelBinarizer()
labels = le.fit_transform(labels)

counts = labels.sum(axis = 0)

# Skew in labeled data
class_totals = labels.sum(axis = 0)

class_weight = {}

# Loop over classes and calculate class weights
for i in range(0, len(class_totals)):
  class_weight[i] = class_totals.max() / class_totals[i]

# Split into train and test values
x_train, x_test, y_train, y_test = train_test_split(
    images,
    labels,
    test_size = 0.25,
    stratify = labels,
    random_state = 42
)

# Data augmentation to improve results
aug = ImageDataGenerator(
    rotation_range = 10,
    zoom_range = 0.05,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.15,
    horizontal_flip = False,
    fill_mode = "nearest"
)

class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red = False, reg = 0.0001, bnEps = 2e-5, bnMom = 0.9):
        # Shortcut: initialize as input data
        shortcut = data

        # First block of ResNet module is 1x1 CONVs
        bn1 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(data)
        act1 = Activation("relu")(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias = False, kernel_regularizer = regularizers.L2(reg))(act1)

        # Second block of ResNet module is 3x3 CONVs
        bn2 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides = stride, padding = "same", use_bias = False, kernel_regularizer = regularizers.L2(reg))(act2)

        # Third block of ResNet module is 1x1 CONVs
        bn3 = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias = False, kernel_regularizer = regularizers.L2(reg))(act3)

        # To reduce spatial size, apply CONV layer to shortcut
        if red:
            shortcut = Conv2D(K, (1, 1), strides = stride, use_bias = False, kernel_regularizer = regularizers.L2(reg))(act1)

        # Add shortcut and final CONV
        final = add([conv3, shortcut])

        return final

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg = 0.0001, bnEps = 2e-5, bnMom = 0.9, dataset = "cifar"):
        # Initialize input with channel last and channel dimensions
        input_shape = (height, width, depth)
        chanDim = -1

        # Channel first, update shape
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1

        # Set input and apply BatchNormalization
        inputs = Input(shape = input_shape)
        x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(inputs)

        # Check dataset
        # cifar
        if dataset == "cifar":
            # Apply single CONV layer
            x = Conv2D(filters[0], (3, 3), use_bias = False, padding = "same", kernel_regularizer = regularizers.L2(reg))(x)

        # tiny imagenet
        elif dataset == "tiny_imagenet":
            # CONV, BN, ACT, POOL to reduce spatial size
            x = Conv2D(filters[0], (5, 5), use_bias = False, padding = "same", kernel_regularizer = regularizers.L2(reg))(x)
            x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(x)
            x = Activation("relu")(x)
            x = ZeroPadding2D((1, 1))(x)
            x = MaxPooling2D((3, 3), strides = (2, 2))(x)

        for i in range(0, len(stages)):
            # Initialize stride and apply residual module to reduce spatial size of input volume
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i + 1], stride, chanDim, red = True, bnEps = bnEps, bnMom = bnMom)

            # Loop through layers in stage
            for j in range(0, stages[i] - 1):
                # Apply ResNet module
                x = ResNet.residual_module(x, filters[i + 1], (1, 1), chanDim, bnEps = bnEps, bnMom = bnMom)

        # Apply BN, ACT, POOL
        x = BatchNormalization(axis = chanDim, epsilon = bnEps, momentum = bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8, 8))(x)

        # Softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer = regularizers.L2(reg))(x)
        x = Activation("softmax")(x)

        # Create model
        model = Model(inputs, x, name = "resnet")

        return model
    
EPOCHS = 50
INIT_LR = 1e-1 # Learning rate
BS = 128 # Batch size

# Stochastic gradient descent optimizer
opt = tf.keras.optimizers.SGD(learning_rate = INIT_LR, weight_decay = INIT_LR / BS)

model = ResNet.build(32, 32, 1, len(le.classes_), (3, 3, 3), (64, 64, 128, 256), reg = 0.0005)

model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics = ["accuracy"])

history = model.fit(
    aug.flow(x_train, y_train, batch_size = BS),
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