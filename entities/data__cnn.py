class DataCnn:

    def __init__(self, image_path, label):
        self.__image = image_path
        self.__label = label
    
    @property
    def image(self):
        return self.__image
    
    @property
    def label(self):
        return self.__label