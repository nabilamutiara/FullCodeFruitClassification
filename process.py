import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


train_dir = '/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/training'
validation_dir = '/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/validation'
test_dir = '/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/testing'


img_height, img_width = 177, 177
batch_size = 32


train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2, rotation_range=20)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                             target_size=(img_height, img_width),
                                                             batch_size=batch_size,
                                                             class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                 target_size=(img_height, img_width),
                                                 batch_size=batch_size,
                                                 class_mode='categorical')


def SimpleNet2D():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_generator.class_indices), activation='softmax'))
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = SimpleNet2D()


model.fit(train_generator,
          epochs=20,
          validation_data=validation_generator)


test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy: {test_accuracy * 100:.2f}%')


model.save('/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/process10.h5')


def classify_image(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    predictions = model.predict(img_array)
    
    
    class_labels = list(train_generator.class_indices.keys())
    probabilities = predictions[0]
    
    
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_labels = [class_labels[i] for i in sorted_indices]
    sorted_probabilities = [probabilities[i] for i in sorted_indices]

    
    for label, prob in zip(sorted_labels, sorted_probabilities):
        print(f'{label.upper()} {prob * 100:.2f}%')


image_path = '/Users/nabilamutiara/Downloads/TugasFinalDeepLearning/630d7ae5d041f.jpg'  # Replace with the actual image path
classify_image(image_path)
