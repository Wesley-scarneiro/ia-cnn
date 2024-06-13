from entities.data__cnn import DataCnn
import os

def open_file(file_path):
    with open(file_path, 'r') as file:
        return file.read().splitlines()

def create_data_cnn(labels_path, dataset_path) -> list[DataCnn]:
    labels = open_file(labels_path)
    images_name = sorted(os.listdir(dataset_path), key=lambda x: int(x.split('.')[0]))
    data_cnn = [DataCnn(image, label) for image, label in zip(images_name, labels[:len(images_name)])]
    return data_cnn

