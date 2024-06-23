import os
import numpy as np

# Abre um arquivo de texto
def open_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

# Cria um array numpy com os caminhos das imagens de forma ordenada e salva em um arquivo .npy
def _create_images_array_numpy(dataset_path, destination_path):
    images = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0]))
    images = [os.path.join(dataset_path, image) for image in images]
    np.save(os.path.join(destination_path, 'images.npy'), np.array(images))

# Cria um array numpy com os labels convertidos para inteiros e salva em um arquivo .npy
def _create_labels_array_numpy(labels_path, destination_path):
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    labels = [letters.index(char) for char in open_file(labels_path)]
    np.save(os.path.join(destination_path, 'labels.npy'), np.array(labels))

