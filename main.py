from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from data.data_cnn import DataCnn
from functions.image_processing import process_images
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Converte rótulos para categóricos (one-hot encoding) com número de classes definida para 26
def convert_labels(train_labels, val_labels, test_labels) -> DataCnn:
    num_classes = 26
    train_labels = to_categorical(train_labels, num_classes)
    val_labels = to_categorical(val_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)
    return train_labels, val_labels, test_labels

# Carregamento, divisão e pré-processamento dos dados que serão utilizados no modelo
def load_data_cnn(images_numpy, labels_numpy) -> DataCnn:
    image_paths = np.load('data\images.npy')
    labels = np.load('data\labels.npy')
    train_images, test_images, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
    train_images, val_images, test_images = process_images(train_images, val_images, test_images)
    train_labels, val_labels, test_labels = convert_labels(train_labels, val_labels, test_labels)
    return DataCnn(train_images, train_labels, val_images, val_labels, test_images, test_labels)

# Cria um modelo CNN para treinamento
def create_model_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(10, 12, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(26, activation='softmax'))

    sgd = SGD(learning_rate=0.1)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Calcula e exibe a matriz de confusão
def plot_confusion_matrix(model, test_images, test_labels, accuracy, path_dir):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    predict = model.predict(test_images)
    predict_classes = np.argmax(predict, axis=1)
    predict_true = np.argmax(test_labels, axis=1)
    conf_matrix = confusion_matrix(predict_true, predict_classes)

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(alphabet), yticklabels=list(alphabet))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {accuracy:.2f}')
    plt.savefig(os.path.join(path_dir, 'confusion_matrix'))

# Treina e avalia o modelo
def train_cnn(images_numpy, labels_numpy, epochs, model, save_path):
    data = load_data_cnn(images_numpy, labels_numpy)
    model.fit(data.train_images, data.train_labels, 
              epochs=epochs, batch_size=32, 
              validation_data=(data.val_images, data.val_labels))
    
    test_loss, test_accuracy = model.evaluate(data.test_images, data.test_labels)
    path_dir = os.path.join(save_path, f'epochs_{epochs}')
    model.save(os.path.join(path_dir, 'cnn_model.h5'))
    print(f'Test accuracy: {test_accuracy:.2f}\nTest loss: {test_loss}')
    plot_confusion_matrix(model, data.test_images, data.test_labels, test_accuracy, path_dir)

def main():
    save_path = 'cnn_models'
    for epoch in range(10, 101, 10):
        model = create_model_cnn()
        train_cnn('data\images.npy', 'data\labels.npy', epoch, model, save_path)

if __name__ == '__main__':
    main()