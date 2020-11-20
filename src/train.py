import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D, MaxPool2D
from keras.callbacks import CSVLogger
from tensorflow.keras import callbacks
from livelossplot import PlotLossesKeras
import numpy as np

TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_FILE = "cats_vs_dogs.h5"


TRAINING_LOGS_FILE = "training_logs.csv"
MODEL_SUMMARY_FILE = "model_summary.txt"
MODEL_FILE = "cats_vs_dogs.h5"

# Data
path = "/Users/shenjianan/CNN_Touch_Tracking/dataset"
test_path = "/Users/shenjianan/CNN_Touch_Tracking/test"
training_data_dir = path + "training" # 10 000 * 2
validation_data_dir = path + "validation" # 2 500 * 2
test_data_dir = path + "test" # 12 500

# Hyperparams
IMAGE_SIZE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 30

input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 3)

# Data augmentation
data_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True)

train_generator = data_generator.flow_from_directory(path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=4,
                                               class_mode='categorical', subset='training', classes=['me', 'others'])
validation_generator = data_generator.flow_from_directory(path,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               batch_size=4,
                                               class_mode='categorical', subset='validation', classes=['me', 'others'])

test_generator = data_generator.flow_from_directory(test_path, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=1, class_mode='categorical', classes=['me', 'others'], shuffle=False)

print(train_generator.filenames)
print(validation_generator.filenames)

# # CNN Model 5 (https://towardsdatascience.com/image-classifier-cats-vs-dogs-with-convolutional-neural-networks-cnns-and-google-colabs-4e9af21ae7a8)
# model = Sequential()

# model.add(Conv2D(32, 3, 3, padding='same', input_shape=input_shape, activation='relu'))
# model.add(Conv2D(32, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, 3, 3, padding='same', activation='relu'))
# model.add(Conv2D(64, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, 3, 3, padding='same', activation='relu'))
# model.add(Conv2D(128, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, 3, 3, padding='same', activation='relu'))
# model.add(Conv2D(256, 3, 3, padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#             optimizer=RMSprop(lr=0.0001),
#             metrics=['accuracy'])

# # with open(MODEL_SUMMARY_FILE,"w") as fh:
# #     model.summary(print_fn=lambda line: fh.write(line + "\n"))

model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dense(units=2, activation='softmax'),
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

model.fit(x=train_generator,
          steps_per_epoch=len(train_generator),
          validation_data=validation_generator,
          validation_steps=len(validation_generator),
          epochs=15,
          verbose=2,
          callbacks=[PlotLossesKeras()]
)

predictions = model.predict(x=test_generator, steps=len(test_generator), verbose=2)
p = np.round(predictions)

predict_result = [0 if i[0] == 1.0 else 1 for i in p]
print(predict_result)
print(test_generator.classes)