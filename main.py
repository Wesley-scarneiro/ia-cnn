from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurações iniciais da CNN
def initialize_cnn():
    classifier = Sequential()
    classifier.add(Input(shape=(10, 12, 1)))  # Adiciona explicitamente uma camada de entrada

    # Primeira camada
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda camada
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())

    # Fully connected layers
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=26, activation='softmax'))  # 26 unidades para 26 classes

    # Compilação do modelo
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

def images_train_validation():
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Pré-processamento das imagens de treino e validação
    training_set = train_datagen.flow_from_directory('data/dataset_train',
                                                     target_size=(10, 12), 
                                                     batch_size=32,
                                                     class_mode='categorical')

    validation_set = validation_datagen.flow_from_directory('data/dataset_validation',
                                                            target_size=(10, 12), 
                                                            batch_size=32,
                                                            class_mode='categorical')
    return training_set, validation_set

# classifier = initialize_cnn()
# training_set, validation_set = images_train_validation()

# # Executando o treinamento
# classifier.fit(training_set,
#                steps_per_epoch=100,
#                epochs=5,
#                validation_data=validation_set,
#                validation_steps=100)

import os

def organize_data(path: str):
    images = []
    