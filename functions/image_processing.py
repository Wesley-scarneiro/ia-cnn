import cv2
import numpy as np

'''
    Carrega e pré-processa as imagens
        - cv2.imread: carrega em escala de cinza
        - cv2.resize: redimensiona para 12x10
        - img.astype: normaliza para o intervalo [0, 1]
        - reshape(-1, 10, 12, 1): redimensiona as imagens para adicionar a dimensão do canal
'''
def _load_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (12, 10))
        img = img.astype('float32') / 255.0
        images.append(img)
    images = np.array(images)
    images.reshape(-1, 10, 12, 1)
    return images

def process_images(train_images, val_imagens, test_images):
    train_images = _load_images(train_images)
    val_imagens = _load_images(val_imagens)
    test_images = _load_images(test_images)
    return train_images, val_imagens, test_images