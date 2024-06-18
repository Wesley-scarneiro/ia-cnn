from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD

model = Sequential()

# Camadas convolucionais
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(10, 12, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Camadas densas
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='softmax'))

sgd = SGD(learning_rate=0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
