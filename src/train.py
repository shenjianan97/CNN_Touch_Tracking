from keras_preprocessing.image import image_data_generator
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
from keras.preprocessing import image

class CNN:
    def __init__(self, datasetPath: str, username: str) -> None:
        self.datasetPath = datasetPath
        self.username = username
        self.IMAGE_SIZE = 200
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = self.IMAGE_SIZE, self.IMAGE_SIZE
        self.EPOCHS = 20
        self.BATCH_SIZE = 32
        self.TEST_SIZE = 30
        self.input_shape = (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3)
        self.model = None

    def train(self):
        # Data augmentation
        data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        train_generator = data_generator.flow_from_directory(self.datasetPath,
                                               target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                                               batch_size=4,
                                               class_mode='categorical', subset='training', classes=[self.username, 'others'])
        validation_generator = data_generator.flow_from_directory(self.datasetPath, target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE), batch_size=4, class_mode='categorical', subset='validation', classes=[self.username, 'others'])

        self.model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(self.IMAGE_SIZE,self.IMAGE_SIZE,3)),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=2, activation='softmax'),
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
        )

        self.model.fit(x=train_generator,
                steps_per_epoch=len(train_generator),
                validation_data=validation_generator,
                validation_steps=len(validation_generator),
                epochs=15,
                verbose=2
        )
        print("train finish")
    
    def predict(self, imagePath):
        img = image.load_img(imagePath, target_size=(self.IMAGE_SIZE, self.IMAGE_SIZE))
        input_array = image.img_to_array(img)
        input_array = input_array/255
        input_array = np.expand_dims(input_array, axis = 0)
        prediction = self.model.predict(input_array, verbose=2)
        return prediction

# predictions = model.predict(x=test_generator, steps=len(test_generator), verbose=2)
# p = np.round(predictions)

# predict_result = [0 if i[0] == 1.0 else 1 for i in p]
# print(predict_result)
# print(test_generator.classes)
if __name__ == '__main__':
    test = CNN("../dataset", "me")
    test.train()
    print(test.predict("swipe1605823125802.jpg"))
    print(test.predict("swipe1605823153763.jpg"))
    print(test.predict("swipe1605819443078.jpg"))